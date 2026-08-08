#!/usr/bin/env python3
import os
import sys

# Setting in case we run on macOS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import warnings
# Suppress known noisy library warnings before any imports that trigger them.
# pynvml FutureWarning fires at torch.cuda import time; max_length UserWarning
# fires at model inference time — both are non-actionable for our users.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*model-agnostic default.*max_length.*")

import asyncio
import argparse
import logging
from pathlib import Path
# PipelineConfiguration/SeedConfiguration/Pipeline/PipelineRunner all live behind
# pipeline.pipeline, which imports every stage class (torch/transformers/diffusers/
# SAM2/...) at module scope. `local` mode only serves a pre-built .frame archive's
# files -- it never touches any of that -- so those imports are deferred into the
# handlers that actually need them (handle_server/handle_run/handle_download)
# instead of paid unconditionally here. Same reasoning for the conda-env check
# below: it's only required for commands that actually run the pipeline.
from server.server import SimulationServerConfiguration, SimulationServer

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Commands that actually run pipeline stages (and therefore need the "frame" conda
# env's torch/transformers/diffusers/SAM2/... stack). `local` just serves a pre-built
# .frame archive's files over HTTP/WebSocket -- no pipeline, no GPU, no special env.
_REQUIRES_FRAME_ENV = {"server", "run", "download"}


def _check_conda_env():
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env != "frame":
        print(
            f"Error: wrong conda environment '{env}'. "
            f"Activate 'frame' first:\n  conda activate frame",
            file=sys.stderr,
        )
        sys.exit(1)

def create_parser():
    parser = argparse.ArgumentParser(
        description="Generate immersive scenes from an image"
    )

    parser.add_argument(
        "--seed",
        type=str,
        action="append",
        metavar="VALUE|STAGE:VALUE",
        help=(
            "Random seed for reproducibility. Pass a single integer to seed all stages "
            "identically, or one or more STAGE:VALUE pairs to seed stages individually. "
            "A global seed is always generated and logged even if not supplied."
        ),
    )
    _log_group = parser.add_mutually_exclusive_group()
    _log_group.add_argument(
        "-v", "--verbose",
        dest="log_mode",
        action="store_const",
        const="verbose",
        help="Print all logs to the terminal with Rich formatting (for debugging)",
    )
    _log_group.add_argument(
        "--plain",
        dest="log_mode",
        action="store_const",
        const="plain",
        help="Print logs as plain timestamped lines (used by server mode)",
    )
    _log_group.add_argument(
        "--log-mode",
        dest="log_mode",
        choices=["panel", "plain", "verbose"],
        metavar="MODE",
        help="Logging mode: panel (default), plain, or verbose",
    )
    parser.set_defaults(log_mode="panel")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # server
    server_parser = subparsers.add_parser(
        "server",
        help="Start the generation server"
    )
    server_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the server"
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on"
    )
    server_parser.add_argument(
        "--asset-port",
        type=int,
        default=3000,
        help="Port to run the asset server on"
    )
    server_parser.add_argument(
        '-d',
        '--debug',
        help="Saves intermediate files for debugg",
        default=True,
        type=bool
    )
    server_parser.add_argument(
        '--debug-archive',
        help="Also write a .debug.frame archive containing intermediate build files",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    server_parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="Output directory"
    )
    server_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline configuration YAML (default: config.yaml)"
    )
    server_parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input image to generate from (default: samples/Mount Rainier.jpg)"
    )
    # Same flag `run` already has, and for the same reason -- but this is the mode that
    # needs it most. Stage caching is decided by has_expected_output() alone; nothing
    # hashes a stage's configuration, so editing a parameter in config.yaml leaves the
    # cached output looking perfectly valid and the stage is skipped. Without this the
    # only way to apply a config change over a warm cache in server/remote mode was to
    # clear the whole output directory and regenerate the panorama and terrain too.
    server_parser.add_argument(
        "--rerun",
        type=str,
        action="append",
        metavar="STAGE[,STAGE...]",
        help=(
            "Purge cached output for the given stage(s) so they are treated as dirty "
            "and re-run (along with all downstream stages). Accepts a comma-separated "
            "list of stage names or output paths. May be repeated."
        ),
    )

    # run
    run_parser = subparsers.add_parser(
        "run",
        help="Run pipeline on an image"
    )
    run_parser.add_argument(
        "input",
        type=str,
        default="",
        help="Input image or directory of images"
    )
    run_parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="Output directory"
    )
    run_parser.add_argument(
        '-d',
        '--debug',
        help="Saves intermediate files for debugg",
        default=True,
        type=bool
    )
    run_parser.add_argument(
        '--debug-archive',
        help="Also write a .debug.frame archive containing intermediate build files",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline configuration YAML (default: config.yaml)"
    )
    run_parser.add_argument(
        "--rerun",
        type=str,
        action="append",
        metavar="STAGE[,STAGE...]",
        help=(
            "Purge cached output for the given stage(s) so they are treated as dirty "
            "and re-run (along with all downstream stages). Accepts a comma-separated "
            "list of stage names or output paths. May be repeated."
        ),
    )

    # local
    local_parser = subparsers.add_parser(
        "local",
        help="Serve a .frame archive as a local scene server (no pipeline required)"
    )
    local_parser.add_argument(
        "archive",
        type=str,
        help="Path to the .frame archive produced by 'run'"
    )
    local_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the server"
    )
    local_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the WebSocket server on"
    )
    local_parser.add_argument(
        "--asset-port",
        type=int,
        default=3000,
        help="Port to run the asset server on"
    )

    # download
    download_parser = subparsers.add_parser(
        "download",
        help="Download all the models needed for the pipeline"
    )
    download_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline configuration YAML (default: config.yaml)"
    )

    return parser

def _parse_seeds(seed_args: list[str] | None):
    from pipeline.pipeline import SeedConfiguration
    if not seed_args:
        return SeedConfiguration()

    global_seed = None
    stage_seeds: dict[str, int] = {}

    for arg in seed_args:
        if ":" in arg:
            stage, _, raw = arg.partition(":")
            stage_seeds[stage.strip()] = int(raw.strip())
        else:
            if global_seed is not None:
                raise ValueError(
                    f"Multiple bare seed values supplied. "
                    f"Use STAGE:VALUE pairs for per-stage seeds."
                )
            global_seed = int(arg)

    return SeedConfiguration(global_seed=global_seed, stage_seeds=stage_seeds)


def _parse_rerun(rerun_args: list[str] | None) -> set[str]:
    if not rerun_args:
        return set()
    result = set()
    for arg in rerun_args:
        for part in arg.split(","):
            name = part.strip()
            if name:
                result.add(name)
    return result


def _create_pipeline_config(args):
    from pipeline.pipeline import PipelineConfiguration
    config = PipelineConfiguration(
        output=args.output,
        seeds=_parse_seeds(getattr(args, "seed", None)),
        config_path=getattr(args, "config", DEFAULT_CONFIG_PATH),
        log_mode=getattr(args, "log_mode", "panel"),
        force_stages=_parse_rerun(getattr(args, "rerun", None)),
    )

    config.save_files = args.debug
    config.debug_archive = getattr(args, "debug_archive", False)

    return config

def handle_server(args):
    from pipeline.pipeline import Pipeline
    from util.path_utils import resource_directory

    if not args.seed:
        args.seed = ["1557422243"]

    configuration = _create_pipeline_config(args=args)

    simulation_config = SimulationServerConfiguration()
    simulation_config.log = configuration.log
    simulation_config.address = args.host
    simulation_config.port = args.port
    # --asset-port was parsed but never applied here, so the asset server always
    # bound 3000 no matter what frame.sh forwarded, while the client was told to
    # use the port it asked for.
    simulation_config.asset_port = args.asset_port
    simulation_config.log_mode = configuration.log_mode

    pipeline = Pipeline(
        config=_create_pipeline_config(args=args)
    )

    # Ensure all the models are downloaded
    pipeline.download_models()

    input_path = Path(args.input) if args.input else resource_directory() / "Mount Rainier.jpg"

    # Run the server!
    server = SimulationServer(simulation_config, pipeline, input_path=input_path)
    asyncio.run(server.run())


def handle_run(args):
    from pipeline.pipeline import Pipeline
    from pipeline.pipeline_input import PipelineInput
    from pipeline.pipeline_runner import PipelineRunner

    config = _create_pipeline_config(args=args)
    pipeline = Pipeline(config=config)

    input = PipelineInput(args.input)
    runner = PipelineRunner(pipeline)
    runner.run(input)

    if config.save_files:
        context_dir = pipeline.context_path()
        if context_dir and context_dir.exists():
            from pipeline.archive import create_frame_archive, create_debug_frame_archive
            from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
            stage_order = [s.name for s in pipeline.stages]
            with Progress(SpinnerColumn(), "[progress.description]{task.description}", TimeElapsedColumn()) as progress:
                task = progress.add_task("Writing archive…", total=None)
                archive = create_frame_archive(
                    context_dir=context_dir,
                    input_path=Path(args.input),
                    output_dir=Path(args.output),
                    stage_order=stage_order,
                )
                if config.debug_archive:
                    progress.update(task, description="Writing debug archive…")
                    contexts = [
                        (item, item_context_dir)
                        for item, item_context_dir in runner.processed
                        if item_context_dir and item_context_dir.exists()
                    ]
                    debug_archive = create_debug_frame_archive(
                        contexts=contexts,
                        input_path=Path(args.input),
                        output_dir=Path(args.output),
                        stage_order=stage_order,
                        log_paths=runner.log_paths(),
                    )
            print(f"Archive: {archive}")
            if config.debug_archive:
                print(f"Debug archive: {debug_archive}")


def handle_local(args):
    import logging
    import tempfile
    from pipeline.archive import load_frame_archive

    archive_path = Path(args.archive)
    if not archive_path.exists():
        print(f"Error: archive not found: {archive_path}")
        return

    log = logging.getLogger("pipeline")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
        ))
        log.addHandler(handler)
    log.setLevel(logging.INFO)

    log.info(f"Loading archive: {archive_path.name}")

    with tempfile.TemporaryDirectory(prefix="frame-local-") as tmpdir:
        context, _ = load_frame_archive(archive_path, Path(tmpdir))

        asset_dir = Path(tmpdir) / "assets"
        asset_dir.mkdir(exist_ok=True)

        sim_config = SimulationServerConfiguration()
        sim_config.log = log
        sim_config.address = args.host
        sim_config.port = args.port
        sim_config.asset_port = args.asset_port
        # `local` builds its own plain StreamHandler logger above rather than a
        # PipelineConfiguration, so every log record already reaches the terminal;
        # a live bar on top of that would fight with them.
        sim_config.log_mode = "plain"

        server = SimulationServer(sim_config, pipeline=None, context=context, asset_dir=asset_dir)
        asyncio.run(server.run())

def handle_download(args):
    from pipeline.pipeline import Pipeline, PipelineConfiguration

    config = PipelineConfiguration(
        output=None,
        config_path=getattr(args, "config", DEFAULT_CONFIG_PATH),
        log_mode=getattr(args, "log_mode", "panel"),
    )

    pipeline = Pipeline(
        config=config
    )

    pipeline.download_models()

def main():
    parser = create_parser()
    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(f"{e}")
        return

    if args.command in _REQUIRES_FRAME_ENV:
        _check_conda_env()
        if args.log_mode != "verbose":
            import transformers
            import diffusers
            transformers.logging.set_verbosity_error()
            diffusers.logging.set_verbosity_error()
            from huggingface_hub import utils as hf_utils
            hf_utils.disable_progress_bars()

    try:
        if args.command == "server":
            handle_server(args)
        elif args.command == "run":
            handle_run(args)
        elif args.command == "local":
            handle_local(args)
        elif args.command == "download":
            handle_download(args)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()