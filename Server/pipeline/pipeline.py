from pathlib import Path
import torch
from typing import Optional
import logging
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler

from pipeline.segmentation.segmentation import SegementationStage
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext

class PipelineConfiguration:
    input: Path
    output: Path
    input: Path

    def __init__(self, input: str, output: str):
        self.input = Path(input)
        self.output = Path(output)
        self.temp = Path(output + "/build")

        self.output.mkdir(parents=True, exist_ok=True)
        self.temp.mkdir(parents=True, exist_ok=True)    

        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.bfloat16
        elif torch.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float
        else:
            self.device = "cpu"
            self.torch_dtype = torch.bfloat16

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        self.log = logging.getLogger("rich")

    def stage_config(self, name: str, input: Optional[Path] = None) -> PipelineStageConfiguration:
        if input is None:
            target_path = self.input
        else:
            target_path = input

        new_config = PipelineStageConfiguration(
            name=name, 
            input=target_path, 
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
            SegementationStage(config=config.stage_config("Object Segementation"))
        ]

    def log_info(self, msg):
        self.config.log.info(msg)

    def run(self):
        self.log_info(f"Running with input: {self.config.input}")
        context = PipelineContext(self.config.input)

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Processing...", total=len(self.stages))

            for stage in self.stages:
                self.log_info(f"Handling stagge {stage.name}")   # scrolls normally above the bar
                stage._set_progress(progress)

                context = stage.run(context)

                stage.clean_up()
                progress.advance(task)

        context.save(self.config.output)
