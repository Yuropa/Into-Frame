import torch
from typing import Optional
from logging import Logger
from pathlib import Path
from rich.progress import Progress
from pipeline.pipeline_context import PipelineContext

class PipelineStageConfiguration:
    input: Path
    output: Path
    temp: Path

    def __init__(
            self,
            name: str, 
            input: Path, 
            output_root: Path, 
            temp: Path, 
            device: str, 
            torch_dtype: any,
            log: Logger
            ):
        self.name = name
        self.input = input
        self.output = output_root / name
        self.temp = temp / name

        self.output.mkdir(parents=True, exist_ok=True)
        self.temp.mkdir(parents=True, exist_ok=True)    

        self.device = device
        self.torch_dtype = torch_dtype
        self.log = log

class PipelineStage:
    def __init__(self, config: PipelineStageConfiguration) -> None:
        self.name = config.name
        self.config = config
        self.device = config.device
        self.torch_dtype = config.torch_dtype

    def model_names(self) -> list[str]:
        return []

    def clean_up(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()

    def log_info(self, message):
        self.config.log.info(message)

    def log_warning(self, message):
        self.config.log.warning(message)

    def log_error(self, message):
        self.config.log.error(message)

    def run(self, input: PipelineContext) -> PipelineContext:
        return input

    def create_progress(self, count: int, label: Optional[str] = None):
        if label is None:
            label = self.name

        return self.progress.add_task("  " + label, total=count)

    def _set_progress(self, progress: Progress):
        self.progress = progress

    
    