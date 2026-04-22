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
from pipeline.scene_generation.generation import SceneGenerationStage
from pipeline.model_generation.generation import ModelGenerationStage
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.device_utils import preferred_device

class PipelineConfiguration:
    output: Optional[Path]
    save_files: bool = False

    def __init__(self, output: Optional[str]):
        if output is not None:
            self.output = Path(output)
            self.temp = Path(output + "/build")

            self.output.mkdir(parents=True, exist_ok=True)
            self._clear_directory(self.output)

            self.temp.mkdir(parents=True, exist_ok=True) 
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

    def _clear_directory(self, path: Path):
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    def stage_config(self, name: str) -> PipelineStageConfiguration:
        new_config = PipelineStageConfiguration(
            name=name,
            output_root=self.output, 
            temp=self.temp, 
            device=self.device, 
            torch_dtype=self.torch_dtype,
            log=self.log
        )

        return new_config

class Pipeline:
    stages: list[PipelineStage]

    def __init__(self, config: PipelineConfiguration):
        self.config = config
        self.device = config.device
        self.torch_dtype = config.torch_dtype

        self.stages = [
            SegmentationStage(config=config.stage_config("Object Segementation")),
            # SupersamplingStage(config=config.stage_config("Supersampling")),
            DepthStage(config=config.stage_config("Depth Generation")),
            ModelGenerationStage(config=config.stage_config("Mesh Generation")),
            SceneGenerationStage(config=config.stage_config("Scene Generation"))
        ]

    def log_info(self, msg):
        self.config.log.info(msg)

    def set_input(self, input):
        self.input = input

    def run(self, progress_queue: Optional[queue.SimpleQueue] = None) -> PipelineContext:
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


    def _run_pipeline(self, progress_queue: Optional[queue.SimpleQueue]) -> PipelineContext:
        self.log_info(f"Running with input: {self.input}")
        context = PipelineContext()
        context.add_image(ContextKey.INPUT, self.input)

        current_step = ""
        current_step_index = 0
        def post_progress():
                if progress_queue is not None:
                    progress_queue.put({"step": current_step, "percent": current_step_index / float(len(self.stages))})

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Processing...", total=len(self.stages))

            for stage in self.stages:
                try:
                    self.log_info(f"Handling stage: {stage.name}")
                    stage._set_progress(progress, task)

                    current_step = stage.name
                    post_progress()

                    context.push_stage(stage.name)
                    context = stage.run(context)
                    context.pop_stage()

                    stage.log_memory_usage()
                    stage.clean_up()
                    progress.advance(task)
            
                    current_step_index += 1
                    post_progress()
                except RuntimeError as e:
                    self._print_total_allocations()
                    raise

        if self.config.save_files:
            if self.config.output is not None:
                context.save(path=self.config.output)

        return context
