#!/usr/bin/env python3
import os
# Setting in case we run on macOS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import asyncio
import argparse
from pathlib import Path
from pipeline.pipeline import Pipeline, PipelineConfiguration, SeedConfiguration, check_conda_env
from pipeline.pipeline_input import PipelineInput
from pipeline.pipeline_runner import PipelineRunner
from server.server import SimulationServerConfiguration, SimulationServer

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

def create_parser():
    parser = argparse.ArgumentParser(
        description="Generate immersive scenes from an image"
    )

    parser.add_argument(
        "--env",
        type=str,
        default="frame",
        help="Expected conda environment name (default: frame)"
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
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Print logs to the terminal in addition to the log file (default: file only)",
    )

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
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline configuration YAML (default: config.yaml)"
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

def _parse_seeds(seed_args: list[str] | None) -> SeedConfiguration:
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


def _create_pipeline_config(args):
    config = PipelineConfiguration(
        output=args.output,
        seeds=_parse_seeds(getattr(args, "seed", None)),
        config_path=getattr(args, "config", DEFAULT_CONFIG_PATH),
        verbose=getattr(args, "verbose", False),
    )

    config.save_files = args.debug

    return config

def handle_server(args):
    configuration = _create_pipeline_config(args=args)

    simulation_config = SimulationServerConfiguration()
    simulation_config.log = configuration.log
    simulation_config.address = args.host
    simulation_config.port = args.port

    pipeline = Pipeline(
        config=_create_pipeline_config(args=args)
    )

    # Ensure all the models are downloaded
    pipeline.download_models()

    # Run the server!
    server = SimulationServer(simulation_config, pipeline)
    asyncio.run(server.run())


def handle_run(args):
    pipeline = Pipeline(
        config=_create_pipeline_config(args=args)
    )

    input = PipelineInput(args.input)
    runner = PipelineRunner(pipeline)
    runner.run(input)

def handle_download(args):
    config = PipelineConfiguration(
        output=None,
        config_path=getattr(args, "config", DEFAULT_CONFIG_PATH),
        verbose=getattr(args, "verbose", False),
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

    check_conda_env(args.env)

    if args.command == "server":
        handle_server(args)
    elif args.command == "run":
        handle_run(args)
    elif args.command == "download":
        handle_download(args)


if __name__ == "__main__":
    main()