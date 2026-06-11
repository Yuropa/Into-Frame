from pathlib import Path
import os
import random
import sys
import torch
import yaml
from typing import Optional
import logging
import shutil
import queue
from collections import deque
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Group as RenderGroup
from huggingface_hub import snapshot_download

from pipeline.segmentation.segmentation import SegmentationStage
from pipeline.supersampling.supersampling import SupersamplingStage
from pipeline.depth.depth import DepthStage
from pipeline.panorama.panorama import PanoramaStage
from pipeline.panorama.panorama_to_cubemap import PanoramaToCubemapStage
from pipeline.scene_generation.generation import SceneGenerationStage
from pipeline.model_generation.generation import ModelGenerationStage
from pipeline.foreground_inpainting.generation import ForegroundInpainting
from pipeline.captioning.captioning import CaptioningStage
from pipeline.heightmap.heightmap import HeightMapStage, HeightMapConfiguration
from pipeline.panorama_depth.depth import PanoramaDepthStage
from pipeline.panorama_inpainting.generation import PanoramaInpaintingStage
from pipeline.panorama_object_classification.classification import PanoramaObjectClassificationStage
from pipeline.object_typing.object_typing import ObjectTypingStage
from pipeline.panorama_asset_generation.generation import PanoramaAssetGenerationStage, PanoramaAssetGenerationConfiguration
from pipeline.lighting.lighting import PanoramaLightingStage
from pipeline.recognize.recognize import RecognizeAnythingStage
from pipeline.terrain.terrain import TerrainMeshStage, TerrainMeshConfiguration
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey, SemanticKeyName
from pipeline.pipeline_context import PipelineContext, ContextKey, ContextKeyName
from pipeline.pipeline_monitor import PipelineMonitor
from pipeline.pipeline_input import PipelineInputItem
from util.device_utils import preferred_device, device_name, DeviceStrategy
from util.image_utils import Image
from util.json_utils import write_json

class _PipelineFilter(logging.Filter):
    """Only passes log records emitted by our own pipeline logger."""
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "pipeline"


class _TailLogPanel(logging.Handler):
    """
    Logging handler that keeps the last N log messages and renders them as a
    fixed-height Rich Panel above the progress bar in non-verbose mode.
    """

    def __init__(self, maxlines: int = 6):
        super().__init__()
        self._lines: deque[str] = deque(maxlen=maxlines)
        self.setFormatter(logging.Formatter("%(message)s"))
        self.addFilter(_PipelineFilter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._lines.append(self.format(record))
        except Exception:
            self.handleError(record)

    def __rich__(self) -> Panel:
        padded = list(self._lines) + [""] * (self._lines.maxlen - len(self._lines))
        text = Text("\n".join(padded), style="dim", no_wrap=True, overflow="ellipsis")
        return Panel(
            text,
            title="[dim]pipeline[/dim]",
            border_style="dim",
            padding=(0, 1),
            height=self._lines.maxlen + 2,
        )


class SeedConfiguration:
    """
    Seed policy for a pipeline run.

    A global seed is always present (randomly generated when not supplied) so
    every run is reproducible by default.  Per-stage overrides take priority over
    the global seed for the named stage; all other stages fall back to global.
    """

    def __init__(
        self,
        global_seed: Optional[int] = None,
        stage_seeds: Optional[dict[str, int]] = None,
    ):
        self.global_seed = global_seed if global_seed is not None else random.randint(0, 2**32 - 1)
        self.stage_seeds: dict[str, int] = stage_seeds or {}

    def seed_for(self, stage_name: str) -> int:
        return self.stage_seeds.get(stage_name, self.global_seed)

    def describe(self) -> str:
        if self.stage_seeds:
            overrides = ", ".join(f"{name}={seed}" for name, seed in self.stage_seeds.items())
            return f"global={self.global_seed} | per-stage: {overrides}"
        return str(self.global_seed)


def check_conda_env(expected: str = "frame"):
    active = os.environ.get("CONDA_DEFAULT_ENV")
    if active != expected:
        print(
            f"Error: wrong conda environment '{active}'. "
            f"Activate '{expected}' first:\n  conda activate {expected}",
            file=sys.stderr,
        )
        sys.exit(1)


def _clear_directory(path: Path):
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

class PipelineConfiguration:
    """Top-level configuration: output/temp paths, device selection, logger, and stage list loaded from YAML."""

    output: Optional[Path]
    save_files: bool = False

    def __init__(
        self,
        output: Optional[str],
        seeds: Optional[SeedConfiguration] = None,
        config_path: Optional[Path] = None,
        verbose: bool = False,
    ):
        if output is not None:
            self.output = Path(output)
            self.temp = Path(output + "/build")

            self.output.mkdir(parents=True, exist_ok=True)

            self.temp.mkdir(parents=True, exist_ok=True)
            _clear_directory(self.temp)
        else:
            self.output = None
            self.temp = None

        self.seeds = seeds if seeds is not None else SeedConfiguration()
        self.verbose = verbose
        self.device, self.torch_dtype = preferred_device(DeviceStrategy.MEMORY)
        self.log = self._configure_logging(verbose)
        self.stages_yaml = self._load_stages_yaml(config_path)

    def _configure_logging(self, verbose: bool) -> logging.Logger:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"pipeline-{timestamp}.log"
        log_path = (self.temp / filename) if self.temp is not None else Path(filename)

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.handlers.clear()

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
        ))
        root.addHandler(file_handler)

        if verbose:
            console_handler = RichHandler(rich_tracebacks=True)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            console_handler.addFilter(_PipelineFilter())
            root.addHandler(console_handler)
        else:
            # Suppress noisy third-party progress bars and HTTP chatter so
            # they don't appear in the tail panel or bleed onto the terminal.
            for noisy in ("httpx", "urllib3", "huggingface_hub", "transformers",
                          "filelock", "diffusers", "accelerate"):
                logging.getLogger(noisy).setLevel(logging.WARNING)
            try:
                from huggingface_hub.utils import disable_progress_bars
                disable_progress_bars()
            except Exception:
                pass

        return logging.getLogger("pipeline")

    def _load_stages_yaml(self, config_path: Optional[Path]) -> list[dict]:
        if config_path is None or not config_path.exists():
            return []
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return data.get("stages", [])

    def stage_config(
        self,
        name: str,
        config_class: type[PipelineStageConfiguration] = PipelineStageConfiguration,
        keys: dict[SemanticKeyName, ContextKeyName] | None = None,
        **kwargs,
    ) -> PipelineStageConfiguration:
        return config_class(
            name=name,
            device=self.device,
            torch_dtype=self.torch_dtype,
            log=self.log,
            keys=keys,
            seed=self.seeds.seed_for(name),
            **kwargs,
        )

STAGE_REGISTRY: dict[str, type[PipelineStage]] = {
    "CaptioningStage": CaptioningStage,
    "DepthStage": DepthStage,
    "PanoramaStage": PanoramaStage,
    "PanoramaToCubemapStage": PanoramaToCubemapStage,
    "PanoramaDepthStage": PanoramaDepthStage,
    "PanoramaLightingStage": PanoramaLightingStage,
    "HeightMapStage": HeightMapStage,
    "TerrainMeshStage": TerrainMeshStage,
    "SegmentationStage": SegmentationStage,
    "SupersamplingStage": SupersamplingStage,
    "SceneGenerationStage": SceneGenerationStage,
    "ModelGenerationStage": ModelGenerationStage,
    "ForegroundInpainting": ForegroundInpainting,
    "PanoramaInpaintingStage": PanoramaInpaintingStage,
    "PanoramaObjectClassificationStage": PanoramaObjectClassificationStage,
    "ObjectTypingStage": ObjectTypingStage,
    "PanoramaAssetGenerationStage": PanoramaAssetGenerationStage,
    "RecognizeAnythingStage": RecognizeAnythingStage,
}


class Pipeline:
    """
    Runs a sequence of PipelineStages against a single input image.

    The stage list is driven by config.yaml (loaded via PipelineConfiguration). Stages are
    executed in order; each receives the shared PipelineContext and may read values written
    by any prior stage. A stage whose has_expected_output() returns True is skipped (cache
    hit). After each stage the context is optionally persisted to disk so a re-run can
    resume from where it left off.
    """

    stages: list[PipelineStage]
    input: PipelineInputItem

    def __init__(self, config: PipelineConfiguration):
        self.config = config
        self.device = config.device
        self.torch_dtype = config.torch_dtype
        self.stages = self._build_stages()
        self.log_info(f"Using device {device_name(self.device)}")

    def _build_stages(self) -> list[PipelineStage]:
        stages = []
        for entry in self.config.stages_yaml:
            if not entry.get("enabled", True):
                continue

            stage_name = entry.get("stage", "")
            if stage_name not in STAGE_REGISTRY:
                raise ValueError(
                    f"Unknown stage '{stage_name}' in config. "
                    f"Available: {sorted(STAGE_REGISTRY)}"
                )

            stage_class = STAGE_REGISTRY[stage_name]
            cfg_class = stage_class.config_class()
            name = entry["name"]

            raw_keys = entry.get("keys")
            keys = {SemanticKey(k): v for k, v in raw_keys.items()} if raw_keys else None

            reserved = {"name", "stage", "enabled", "keys"}
            kwargs = {k: v for k, v in entry.items() if k not in reserved}

            stage_config = self.config.stage_config(name, cfg_class, keys=keys, **kwargs)
            stages.append(stage_class(config=stage_config))

        return stages

    def context_path(self) -> Optional[Path]:
        """Returns the persisted output directory for the last-run input, or None."""
        if not hasattr(self, "input") or self.input is None:
            return None
        if self.config.output is None:
            return None
        return self.config.output / self.input.uuid_string()

    def _create_output_directories(self) -> Optional[Path]:
        input_name = self.input.uuid_string()
        if self.config.output is not None:
            output = self.config.output / input_name
            output.mkdir(parents=True, exist_ok=True)
            return output
        return None

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
            output = self._create_output_directories()
            if output is not None:
                context.save(path=output)


    def _post_progress(self, progress_queue: Optional[queue.SimpleQueue]):
        if progress_queue is not None:
            progress_queue.put({"step": self.current_step, "percent": self.current_step_index / float(len(self.stages))})

    def _run_stage(self, stage: PipelineStage, context: PipelineContext, progress_queue: Optional[queue.SimpleQueue], monitor, progress, task) -> bool:
        output_root = self._create_output_directories()
        stage.set_output(output_root)

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
                    ran = True
                else:
                    self.log_info(f"Skipping cached stage {stage.name}")
                    ran = False

                context.pop_stage()

                self._save_context(context)
                self.current_step_index += 1
                self._post_progress(progress_queue)
                return ran
            except RuntimeError as e:
                self._print_total_allocations()
                raise

    def _run_pipeline(self, progress_queue: Optional[queue.SimpleQueue]) -> PipelineContext:
        self.log_info(f"Seed: {self.config.seeds.describe()}")
        self.log_info(f"Running with input: {self.input}")

        output = self._create_output_directories()
        if output is not None:
            seed_data = {
                "global_seed": self.config.seeds.global_seed,
                "stage_seeds": self.config.seeds.stage_seeds,
            }
            with open(output / "seed.json", "w") as f:
                write_json(seed_data, f)

        context = PipelineContext()
        input_image = self.input.image

        if self.config.output is not None and self.config.output.exists():
            self.log_info("Loading cached content")
            stage_order = [stage.name for stage in self.stages]
            output = self._create_output_directories()
            context.load(output, stage_order)

            orig_input_image = context.image(ContextKey.INPUT)
            if not input_image == orig_input_image:
                self.log_info("New input image, purging stored content")
                _clear_directory(output)
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
            if self.config.verbose:
                progress = Progress(
                    SpinnerColumn(),
                    "[progress.description]{task.description}",
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeElapsedColumn(),
                )
                live_ctx = progress
            else:
                progress = Progress(
                    SpinnerColumn(),
                    "[progress.description]{task.description}",
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeElapsedColumn(),
                    auto_refresh=False,
                )
                tail = _TailLogPanel(maxlines=6)
                logging.getLogger().addHandler(tail)
                live_ctx = Live(RenderGroup(tail, progress), refresh_per_second=4)

            try:
                with live_ctx:
                    task = progress.add_task("Processing...", total=len(self.stages))
                    for stage in self.stages:
                        self._run_stage(
                            stage=stage,
                            context=context,
                            progress_queue=progress_queue,
                            monitor=monitor,
                            progress=progress,
                            task=task,
                        )
            finally:
                if not self.config.verbose:
                    logging.getLogger().removeHandler(tail)

        self._save_context(context)
        monitor.print_summary()

        return context
