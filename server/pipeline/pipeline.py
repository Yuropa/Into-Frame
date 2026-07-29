from pathlib import Path
import io
import os
import random
import sys
import torch
import yaml
from typing import Optional
import logging
import shutil
import queue
import threading
from collections import deque
from rich.console import Console as RichConsole
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TextColumn
from rich.logging import RichHandler
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Group as RenderGroup
from huggingface_hub import snapshot_download

from pipeline.segmentation.segmentation import SegmentationStage
from pipeline.segmentation.video_object_extraction import VideoObjectExtractionStage
from pipeline.supersampling.supersampling import SupersamplingStage
from pipeline.depth.depth import DepthStage
from pipeline.panorama.panorama import PanoramaStage
from pipeline.panorama.panorama_to_cubemap import PanoramaToCubemapStage
from pipeline.scene_generation.generation import SceneGenerationStage
from pipeline.foreground_inpainting.generation import ForegroundInpainting
from pipeline.captioning.captioning import CaptioningStage
from pipeline.heightmap.heightmap import HeightMapStage, HeightMapConfiguration
from pipeline.panorama_depth.depth import PanoramaDepthStage
from pipeline.panorama_depth.calibration import PanoramaDepthCalibrationStage
from pipeline.panorama_inpainting.generation import PanoramaInpaintingStage
from pipeline.panorama_foreground_inpainting.generation import PanoramaForegroundInpaintingStage
from pipeline.panorama_lora_correction.generation import PanoramaLoraCorrectionStage
from pipeline.panorama_depth_patch.generation import PanoramaDepthPatchStage
from pipeline.panorama_object_classification.classification import PanoramaObjectClassificationStage
from pipeline.object_typing.object_typing import ObjectTypingStage
from pipeline.panorama_asset_generation.generation import PanoramaAssetGenerationStage, PanoramaAssetGenerationConfiguration
from pipeline.lighting.lighting import PanoramaLightingStage
from pipeline.recognize.recognize import RecognizeAnythingStage
from pipeline.object_correlation.object_correlation import ObjectCorrelationStage
from pipeline.object_detection.object_detection import ObjectDetectionStage
from pipeline.object_instance_refinement.refinement import ObjectInstanceRefinementStage
from pipeline.object_clustering.clustering import ObjectCategoryClusteringStage
from pipeline.object_distribution.object_distribution import ObjectDistributionStage
from pipeline.grass_cover.grass_cover import GrassCoverStage
from pipeline.distribution_synthesis.distribution_synthesis import DistributionSynthesisStage
from pipeline.panorama_segmentation.panorama_segmentation import PanoramaRegionStage
from pipeline.region_map.region_map import RegionMapStage, RegionMapConfiguration
from pipeline.region_map.region_map_refinement import RegionMapRefinementStage, RegionMapRefinementConfiguration
from pipeline.terrain.terrain import TerrainMeshStage, TerrainMeshConfiguration
from pipeline.terrain.terrain_texture_generation import TerrainTextureGenerationStage, TerrainTextureGenerationConfiguration
from pipeline.linear_structures.linear_structures import LinearStructureStage, LinearStructureConfiguration
from pipeline.terrain.terrain_reconstruction import TerrainReconstructionStage, TerrainReconstructionConfiguration
from pipeline.terrain.terrain_noise_refinement import TerrainNoiseRefinementStage
from pipeline.tree_generation.tree_generation import TreeMeshGenerationStage
from pipeline.video_generation.video_generation import VideoGenerationStage
from pipeline.motion_classification.motion_classification import ObjectMotionClassificationStage
from pipeline.motion_classification.category_mesh_rigging import CategoryMeshRiggingStage
from pipeline.motion_classification.scene_animation import SceneAnimationStage
from pipeline.skybox_inpainting.skybox_inpainting import SkyboxInpaintingStage
from pipeline.report.report import ReportGenerationStage
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey, SemanticKeyName
from pipeline.pipeline_context import PipelineContext, ContextKey, ContextKeyName
from pipeline.pipeline_monitor import PipelineMonitor
from pipeline.pipeline_input import PipelineInputItem
from util.device_utils import preferred_device, device_name, DeviceStrategy
from util.image_utils import Image
from util.json_utils import write_json, parse_json

class _PipelineFilter(logging.Filter):
    """Passes log records from the pipeline logger and its children."""
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "pipeline" or record.name.startswith("pipeline.")


class _LogStream:
    """Redirects stray print() calls into the pipeline logger during Live display."""

    def __init__(self, log: logging.Logger, level: int = logging.INFO):
        self._log = log
        self._level = level
        self._buf = ""
        self.encoding = "utf-8"
        self.errors = "replace"

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            data = data.decode("utf-8", errors="replace")
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._log.log(self._level, line)
        return len(data)

    def flush(self):
        if self._buf.strip():
            self._log.log(self._level, self._buf)
            self._buf = ""

    def isatty(self):
        return False

    def fileno(self):
        raise io.UnsupportedOperation("fileno")


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
        # Tracked separately from the resolved values below so a resumed run
        # against an existing cache (see Pipeline._run_pipeline) can tell "the
        # caller actually asked for this seed" apart from "nothing was passed,
        # so a fresh one got minted here to keep this run reproducible" --
        # only the former should ever be compared against a cached seed.json
        # to decide a stage is dirty.
        self.explicit_global_seed = global_seed is not None
        self.explicit_stage_seeds: set[str] = set((stage_seeds or {}).keys())
        self.global_seed = global_seed if global_seed is not None else random.randint(0, 2**32 - 1)
        self.stage_seeds: dict[str, int] = dict(stage_seeds or {})

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
    debug_archive: bool = False

    def __init__(
        self,
        output: Optional[str],
        seeds: Optional[SeedConfiguration] = None,
        config_path: Optional[Path] = None,
        log_mode: str = "panel",
        force_stages: Optional[set[str]] = None,
    ):
        """
        log_mode controls logging output:
          "panel"   — Rich panel UI during pipeline runs, all library noise suppressed.
                      Default for the 'run' command.
          "plain"   — Plain StreamHandler to stderr, noise still suppressed.
                      Default for the 'server' command.
          "verbose" — RichHandler with full output, minimal suppression.
                      For developer debugging (--verbose flag).
        """
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
        self.log_mode = log_mode
        self.force_stages: set[str] = force_stages or set()
        self.device, self.torch_dtype = preferred_device()
        self.log = self._configure_logging(log_mode)
        self.stages_yaml = self._load_stages_yaml(config_path)

    def _suppress_library_noise(self):
        """Silence tqdm bars and chatty third-party loggers."""
        for noisy in ("httpx", "urllib3", "huggingface_hub", "transformers",
                      "filelock", "diffusers", "accelerate"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except Exception:
            pass
        os.environ["TQDM_DISABLE"] = "1"
        try:
            import transformers as _transformers
            _transformers.logging.set_verbosity_error()
        except Exception:
            pass
        try:
            import diffusers as _diffusers
            _diffusers.logging.set_verbosity_error()
        except Exception:
            pass

    def _configure_logging(self, log_mode: str) -> logging.Logger:
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

        if log_mode == "verbose":
            # Developer debugging: Rich console output, no noise suppression.
            console_handler = RichHandler(rich_tracebacks=True)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            console_handler.addFilter(_PipelineFilter())
            root.addHandler(console_handler)
        elif log_mode == "plain":
            # Server mode: plain timestamped lines to stderr, noise suppressed.
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
            ))
            console_handler.addFilter(_PipelineFilter())
            root.addHandler(console_handler)
            self._suppress_library_noise()
        else:
            # Panel mode (default): no console handler — the panel is added
            # temporarily by _run_pipeline for the duration of each run.
            self._suppress_library_noise()

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
    "PanoramaDepthCalibrationStage": PanoramaDepthCalibrationStage,
    "PanoramaLightingStage": PanoramaLightingStage,
    "HeightMapStage": HeightMapStage,
    "TerrainMeshStage": TerrainMeshStage,
    "TerrainTextureGenerationStage": TerrainTextureGenerationStage,
    "SegmentationStage": SegmentationStage,
    "SupersamplingStage": SupersamplingStage,
    "SceneGenerationStage": SceneGenerationStage,
    "ForegroundInpainting": ForegroundInpainting,
    "PanoramaInpaintingStage": PanoramaInpaintingStage,
    "PanoramaForegroundInpaintingStage": PanoramaForegroundInpaintingStage,
    "PanoramaLoraCorrectionStage": PanoramaLoraCorrectionStage,
    "PanoramaDepthPatchStage": PanoramaDepthPatchStage,
    "PanoramaObjectClassificationStage": PanoramaObjectClassificationStage,
    "ObjectTypingStage": ObjectTypingStage,
    "PanoramaAssetGenerationStage": PanoramaAssetGenerationStage,
    "RecognizeAnythingStage": RecognizeAnythingStage,
    "ObjectCorrelationStage": ObjectCorrelationStage,
    "ObjectDetectionStage": ObjectDetectionStage,
    "ObjectInstanceRefinementStage": ObjectInstanceRefinementStage,
    "ObjectCategoryClusteringStage": ObjectCategoryClusteringStage,
    "ObjectDistributionStage": ObjectDistributionStage,
    "DistributionSynthesisStage": DistributionSynthesisStage,
    "GrassCoverStage": GrassCoverStage,
    "LinearStructureStage": LinearStructureStage,
    "TerrainReconstructionStage": TerrainReconstructionStage,
    "TerrainNoiseRefinementStage": TerrainNoiseRefinementStage,
    "PanoramaRegionStage": PanoramaRegionStage,
    "RegionMapStage": RegionMapStage,
    "RegionMapRefinementStage": RegionMapRefinementStage,
    "TreeMeshGenerationStage": TreeMeshGenerationStage,
    "VideoGenerationStage": VideoGenerationStage,
    "VideoObjectExtractionStage": VideoObjectExtractionStage,
    "ObjectMotionClassificationStage": ObjectMotionClassificationStage,
    "CategoryMeshRiggingStage": CategoryMeshRiggingStage,
    "SceneAnimationStage": SceneAnimationStage,
    "SkyboxInpaintingStage": SkyboxInpaintingStage,
    "ReportGenerationStage": ReportGenerationStage,
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

    def log_error(self, msg):
        self.config.log.error(msg)

    def run(self, input: PipelineInputItem, progress_queue: Optional[queue.SimpleQueue] = None) -> PipelineContext:
        self.input = input
        self.download_models()
        return self._run_pipeline(progress_queue)

    def download_models(self):
        all_models = sorted(set(
            model
            for stage in self.stages
            for model in stage.model_names()
        ))

        if not all_models:
            return

        # Runs before the Live UI starts, and panel/plain mode suppress HF's own
        # tqdm bars + drop logging to WARNING (see _suppress_library_noise), so a
        # slow/first-time download would otherwise show nothing at all. A single
        # transient spinner line gives visible progress without that per-file
        # byte-progress spam. Verbose mode already shows HF's raw bars for this,
        # and a second Live region would fight with them, so skip it there.
        if self.config.log_mode == "verbose":
            for model in all_models:
                self.log_info(f"Checking for model: {model}")
                snapshot_download(repo_id=model)
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=RichConsole(),
                transient=True,
            ) as progress:
                task = progress.add_task("Checking models…", total=len(all_models))
                for model in all_models:
                    progress.update(task, description=f"Checking model: {model}")
                    self.log_info(f"Checking for model: {model}")

                    # Most models are already cached and resolve near-instantly;
                    # only nudge toward -v once a single model has stalled for a
                    # while, so the hint doesn't fire on every fast HEAD check.
                    stall_hint = threading.Timer(
                        5.0,
                        lambda m=model: progress.update(
                            task, description=f"Checking model: {m} (re-run with -v to see download progress)"
                        ),
                    )
                    stall_hint.start()
                    try:
                        snapshot_download(repo_id=model)
                    finally:
                        stall_hint.cancel()

                    progress.advance(task)

        self.log_info("All models present")

    def _print_total_allocations(self):
        if torch.backends.mps.is_available():
            self.log_info(f"MPS Allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
            self.log_info(f"MPS Driver: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")
            self.log_info(f"MPS Cap: {torch.mps.recommended_max_memory() / 1e9:.2f} GB")
        elif torch.cuda.is_available():
            self.log_info(f"CUDA Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            self.log_info(f"CUDA Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            self.log_info(f"CUDA Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    def _save_context(self, context: PipelineContext):
        if self.config.save_files:
            self.log_info("Writing context to disk")
            output = self._create_output_directories()
            if output is not None:
                context.save(path=output)


    def _resolve_stage_name(self, spec: str) -> str:
        """Return the stage name from a spec that may be a bare name or a path."""
        return Path(spec).name if ("/" in spec or "\\" in spec) else spec

    def _post_progress(self, progress_queue: Optional[queue.SimpleQueue]):
        if progress_queue is not None:
            progress_queue.put({"step": self.current_step, "percent": self.current_step_index / float(len(self.stages))})

    def _run_stage(self, stage: PipelineStage, context: PipelineContext, progress_queue: Optional[queue.SimpleQueue], monitor, progress, task, force: bool = False) -> bool:
        output_root = self._create_output_directories()
        stage.set_output(output_root)

        with monitor.stage(stage.name):
            try:
                self.log_info(f"Handling stage: {stage.name}")
                stage._set_progress(progress, task)

                self.current_step = stage.name
                self._post_progress(progress_queue)

                context.push_stage(stage.name)
                if force or not stage.has_expected_output(context):
                    try:
                        context = stage.run(context)
                        stage.log_memory_usage()
                    finally:
                        stage.clean_up()
                    ran = True
                else:
                    self.log_info(f"Skipping cached stage {stage.name}")
                    progress.advance(task, 1)
                    ran = False

                context.pop_stage()

                section = stage.contribute_report(context)
                if section is not None:
                    context.add_report_section(section)

                self._save_context(context)
                self.current_step_index += 1
                self._post_progress(progress_queue)
                return ran
            except Exception as e:
                self.log_error(f"Stage '{stage.name}' failed: {type(e).__name__}: {e}")
                self._print_total_allocations()
                raise

    def _run_pipeline(self, progress_queue: Optional[queue.SimpleQueue]) -> PipelineContext:
        info = Text()
        info.append("Image:  ", style="bold cyan")
        info.append(str(self.input.source_path), style="white")
        info.append("\nHash:   ", style="bold cyan")
        info.append(self.input.uuid_string(), style="white")
        info.append("\nSeed:   ", style="bold cyan")
        info.append(self.config.seeds.describe(), style="bright_yellow bold")
        RichConsole().print(Panel(info, title="[bold green]Pipeline State[/bold green]", border_style="green", expand=True, padding=(1, 2)))

        output = self._create_output_directories()

        # Read cached seeds before overwriting so we can detect per-stage changes.
        cached_global_seed: Optional[int] = None
        cached_stage_seeds: dict[str, int] = {}
        if output is not None:
            seed_file = output / "seed.json"
            if seed_file.exists():
                with open(seed_file) as f:
                    old_seed_data = parse_json(f.read())
                cached_global_seed = old_seed_data.get("global_seed")
                cached_stage_seeds = old_seed_data.get("stage_seeds", {})

        # Adopt the cached seed for anything the caller didn't explicitly ask a
        # seed for -- otherwise every run with no --seed flag mints a fresh
        # random global_seed (see SeedConfiguration.__init__), which will
        # essentially never equal cached_global_seed above, so the "seed
        # changed" check a few lines below would fire for the very first
        # stage checked and cascade a forced rerun of the *entire* rest of
        # the pipeline on every single invocation -- regardless of --rerun,
        # regardless of whether anything actually changed. Only a seed the
        # caller genuinely requested should ever be compared against cache.
        if not self.config.seeds.explicit_global_seed and cached_global_seed is not None:
            self.config.seeds.global_seed = cached_global_seed
        for stage_name, cached_seed_value in cached_stage_seeds.items():
            if stage_name not in self.config.seeds.explicit_stage_seeds:
                self.config.seeds.stage_seeds[stage_name] = cached_seed_value

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
            # Full declared order, including currently-disabled stages -- lets
            # load() place a disabled-but-still-cached stage's data at its true
            # relative position instead of wherever filesystem iteration finds it
            # (see PipelineContext.load / _insert_in_declared_order).
            full_stage_order = [entry["name"] for entry in self.config.stages_yaml]
            output = self._create_output_directories()
            context.load(output, stage_order, full_stage_order)

            orig_input_image = context.image(ContextKey.INPUT)
            if not input_image == orig_input_image:
                self.log_info("New input image, purging stored content")
                _clear_directory(output)
                context = PipelineContext()
                context.add_image(ContextKey.INPUT, input_image)
                # Cache was purged — seed comparison no longer meaningful.
                cached_global_seed = None
                cached_stage_seeds = {}
            else:
                self.log_info("Found cached content")
                context.log_state()
        else:
            context.add_image(ContextKey.INPUT, input_image)

        self.current_step = ""
        self.current_step_index = 0

        monitor = PipelineMonitor(interval=0.25)
        _orig_stdout = sys.stdout
        _orig_stderr = sys.stderr
        _redirected = False

        log_mode = self.config.log_mode

        with monitor.stage("Full Pipeline"):
            if log_mode in ("verbose", "plain"):
                # Verbose / plain: plain Rich progress bar, logs already go to
                # terminal via the handler set up in _configure_logging.
                progress = Progress(
                    SpinnerColumn(),
                    "[progress.description]{task.description}",
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeElapsedColumn(),
                )
                live_ctx = progress
            else:
                # Panel mode: pin Rich to the real stdout before redirecting
                # sys.stdout, so stray print() calls route to the logger instead
                # of corrupting the Live display.
                _console = RichConsole(file=_orig_stdout, stderr=False)
                progress = Progress(
                    SpinnerColumn(),
                    "[progress.description]{task.description}",
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeElapsedColumn(),
                    auto_refresh=False,
                    console=_console,
                )
                tail = _TailLogPanel(maxlines=10)
                logging.getLogger().addHandler(tail)
                _hint = Text("  Ctrl+C to exit early", style="dim")
                live_ctx = Live(
                    RenderGroup(tail, progress, _hint),
                    refresh_per_second=4,
                    console=_console,
                )
                sys.stdout = _LogStream(self.config.log, logging.INFO)
                sys.stderr = _LogStream(self.config.log, logging.WARNING)
                _redirected = True

            _pipeline_exc: Optional[BaseException] = None
            try:
                with live_ctx:
                    task = progress.add_task("Processing…", total=len(self.stages))
                    dirty = False
                    for stage in self.stages:
                        cached_seed = cached_stage_seeds.get(stage.name, cached_global_seed)
                        current_seed = self.config.seeds.seed_for(stage.name)
                        seed_changed = cached_global_seed is not None and cached_seed != current_seed
                        stage_forced = self._resolve_stage_name(stage.name) in self.config.force_stages
                        if seed_changed and not dirty:
                            self.log_info(f"Seed changed for '{stage.name}' ({cached_seed} → {current_seed}), forcing rerun")
                        if stage_forced and not dirty:
                            self.log_info(f"Stage '{stage.name}' marked for rerun via --rerun")
                        ran = self._run_stage(
                            stage=stage,
                            context=context,
                            progress_queue=progress_queue,
                            monitor=monitor,
                            progress=progress,
                            task=task,
                            force=dirty or seed_changed or stage_forced,
                        )
                        dirty = dirty or ran
            except BaseException as _exc:
                _pipeline_exc = _exc
                raise
            finally:
                if _redirected:
                    sys.stdout = _orig_stdout
                    sys.stderr = _orig_stderr
                    logging.getLogger().removeHandler(tail)
                    # In panel mode stderr was redirected into the logger, so the
                    # exception would only appear as a silent panel log before the
                    # display collapsed. Now that the real stderr is restored, write
                    # a clear banner so the error is always visible on the terminal.
                    if _pipeline_exc is not None and not isinstance(_pipeline_exc, KeyboardInterrupt):
                        _orig_stderr.write(
                            f"\n\033[1;31m{'─' * 60}\n"
                            f"Pipeline error in '{self.current_step}':\n"
                            f"  {type(_pipeline_exc).__name__}: {_pipeline_exc}\n"
                            f"{'─' * 60}\033[0m\n"
                        )
                        _orig_stderr.flush()

        self._save_context(context)
        monitor.print_summary()

        return context
