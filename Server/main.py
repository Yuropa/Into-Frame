#!/usr/bin/env python3

import argparse
from pipeline.pipeline import Pipeline, PipelineConfiguration

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
        default="127.0.0.1",
        help="Host to bind the server"
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on"
    )
    server_parser.add_argument(
        '-d',
        '--debug',
        help="Saves intermediate files for debugg",
        default=False,
        type=bool
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
        input=args.input,
        output=args.output
    )

    config.save_files = args.debug

    return config

def handle_server(args):
    print(f"Starting server on {args.host}:{args.port}")
    configuration = _create_pipeline_config(args=args)
    # TODO: add actual server logic here


def handle_run(args):
    pipeline = Pipeline(
        config=_create_pipeline_config(args=args)
    )

    pipeline.run()

def handle_download(args):
    config = PipelineConfiguration(
        input=None,
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