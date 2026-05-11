from pathlib import Path
import torch
from typing import Optional
import logging
import shutil
import queue
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
from huggingface_hub import snapshot_download

from pipeline.segmentation.segmentation import SegmentationStage
from pipeline.supersampling.supersampling import SupersamplingStage
from pipeline.depth.depth import DepthStage
from pipeline.panorama.panorama import PanoramaStage
from pipeline.scene_generation.generation import SceneGenerationStage
from pipeline.model_generation.generation import ModelGenerationStage
from pipeline.captioning.captioning import CaptioningStage
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey, SemanticKeyName
from pipeline.pipeline_context import PipelineContext, ContextKey, ContextKeyName
from pipeline.pipeline_monitor import PipelineMonitor
from pipeline.pipeline_input import PipelineInputItem
from util.device_utils import preferred_device, device_name
from util.image_utils import Image

def _clear_directory(path: Path):
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

class PipelineConfiguration:
    output: Optional[Path]
    save_files: bool = False

    def __init__(self, output: Optional[str]):
        if output is not None:
            self.output = Path(output)
            self.temp = Path(output + "/build")

            self.output.mkdir(parents=True, exist_ok=True)

            self.temp.mkdir(parents=True, exist_ok=True) 
            _clear_directory(self.temp)
        else:
            self.output = None
            self.temp = None

        self.device, self.torch_dtype = preferred_device()

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        self.log = logging.getLogger("rich")

    def stage_config(self, name: str, keys: dict[SemanticKeyName, ContextKeyName] | None = None,) -> PipelineStageConfiguration:
        new_config = PipelineStageConfiguration(
            name=name,
            device=self.device, 
            torch_dtype=self.torch_dtype,
            log=self.log,
            keys=keys
        )

        return new_config

class Pipeline:
    stages: list[PipelineStage]
    input: PipelineInputItem

    def __init__(self, config: PipelineConfiguration):
        self.config = config
        self.device = config.device
        self.torch_dtype = config.torch_dtype

        self.stages = [
            SegmentationStage(config=config.stage_config("Object Segementation")),
            CaptioningStage(config=config.stage_config("Captioning")),
            DepthStage(config=config.stage_config("Depth Generation")),
            PanoramaStage(config=config.stage_config("Panorama")),
            DepthStage(config=config.stage_config("Pano Depth", keys={
                SemanticKey.INPUT: ContextKey.PANAORAMA_CUBENAME,
                SemanticKey.OUTPUT: "Panorama Depth"
            })),
            # ModelGenerationStage(config=config.stage_config("Mesh Generation")),
            SceneGenerationStage(config=config.stage_config("Scene Generation"))
        ]

        self.log_info(f"Using device {device_name(self.device)}")

    def _create_output_directories(self) -> tuple[Optional[Path], Optional[Path]]:
        input_name = self.input.uuid_string()
        if self.config.output is not None:
            output = self.config.output / input_name
            output.mkdir(parents=True, exist_ok=True)
        else:
            output = None

        if self.config.temp is not None:
            temp = self.config.temp / input_name
            temp.mkdir(parents=True, exist_ok=True)
        else:
            temp = None

        return output, temp

    def log_info(self, msg):
        self.config.log.info(msg)

    def run(self, input: PipelineInputItem, progress_queue: Optional[queue.SimpleQueue] = None) -> PipelineContext:
        self.input = input
        self.download_models()
        return self._run_pipeline(progress_queue)

    def download_models(self):
        all_models = set()

        for stage in self.stages:
            for model in stage.model_names():
                all_models.add(model)

        for model in all_models:
            self.log_info(f"Checking for model: {model}")
            snapshot_download(repo_id=model)

        self.log_info("All models present")

    def _print_total_allocations(self):
        if torch.backends.mps.is_available():
            print("MPS Allocated:", torch.mps.current_allocated_memory() / 1e9, "GB")
            print("MPS Driver:", torch.mps.driver_allocated_memory() / 1e9, "GB")
            print("MPS Cap:", torch.mps.recommended_max_memory() / 1e9, "GB")
        elif torch.cuda.is_available():
            print("CUDA Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
            print("CUDA Reserved:", torch.cuda.memory_reserved() / 1e9, "GB")
            print("CUDA Max Allocated:", torch.cuda.max_memory_allocated() / 1e9, "GB")

    def _save_context(self, context: PipelineContext):
        if self.config.save_files:
            self.log_info("Writing context to disk")
            output, _ = self._create_output_directories()
            if output is not None:
                context.save(path=output)


    def _post_progress(self, progress_queue: Optional[queue.SimpleQueue]):
        if progress_queue is not None:
            progress_queue.put({"step": self.current_step, "percent": self.current_step_index / float(len(self.stages))})

    def _run_stage(self, stage: PipelineStage, context: PipelineContext, progress_queue: Optional[queue.SimpleQueue], monitor, progress, task):
        output_root, temp_root = self._create_output_directories()
        stage.set_output(output_root, temp_root)
        
        with monitor.stage(stage.name):    
            try:
                self.log_info(f"Handling stage: {stage.name}")
                stage._set_progress(progress, task)

                self.current_step = stage.name
                self._post_progress(progress_queue)

                context.push_stage(stage.name)
                if not stage.has_expected_output(context):
                    context = stage.run(context)
                    stage.log_memory_usage()
                    stage.clean_up()
                else:
                    self.log_info(f"Skipping cached stage {stage.name}")
        
                context.pop_stage()

                self._save_context(context)
                self.current_step_index += 1
                self._post_progress(progress_queue)
            except RuntimeError as e:
                self._print_total_allocations()
                raise

    def _run_pipeline(self, progress_queue: Optional[queue.SimpleQueue]) -> PipelineContext:
        self.log_info(f"Running with input: {self.input}")
        context = PipelineContext()
        input_image = self.input.image

        if self.config.output is not None and self.config.output.exists():
            self.log_info("Loading cached content")
            stage_order = [stage.name for stage in self.stages]
            output, _ = self._create_output_directories()
            context.load(output, stage_order)

            orig_input_image = context.image(ContextKey.INPUT)
            if not input_image == orig_input_image:
                self.log_info("New input image, purging stored content")
                _clear_directory(self.config.output)
                context = PipelineContext()
                context.add_image(ContextKey.INPUT, input_image)
            else:
                self.log_info("Found cached content")
                context.log_state()
        else:
            context.add_image(ContextKey.INPUT, input_image)

        self.current_step = ""
        self.current_step_index = 0

        monitor = PipelineMonitor(interval=0.25)

        with monitor.stage("Full Pipeline"):
            with Progress(
                SpinnerColumn(),
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task("Processing...", total=len(self.stages))

                for stage in self.stages:
                    self._run_stage(
                        stage=stage,
                        context=context,
                        progress_queue=progress_queue,
                        monitor=monitor,
                        progress=progress,
                        task=task
                    )                    

        self._save_context(context)
        monitor.print_summary()

        return context
