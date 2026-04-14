#!/usr/bin/env python3

import asyncio
import argparse
from pipeline.pipeline import Pipeline, PipelineConfiguration
from server.server import SimulationServerConfiguration, SimulationServer

def create_parser():
    parser = argparse.ArgumentParser(
        description="Generate immersive scenes from an image"
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
        default=False,
        type=bool
    )
    server_parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="Output directory"
    )

    # run
    run_parser = subparsers.add_parser(
        "run",
        help="Run pipeline on an image"
    )
    run_parser.add_argument(
        "input",
        type=str,
        help="Input image"
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

    # download
    download_parser = subparsers.add_parser(
        "download",
        help="Download all the models needed for the pipeline"
    )

    return parser

def _create_pipeline_config(args):
    config = PipelineConfiguration(
        output=args.output
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

    pipeline.set_input(args.input)
    pipeline.run()

def handle_download(args):
    config = PipelineConfiguration(
        output=None
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

    if args.command == "server":
        handle_server(args)
    elif args.command == "run":
        handle_run(args)
    elif args.command == "download":
        handle_download(args)


if __name__ == "__main__":
    main()