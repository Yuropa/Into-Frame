import torch
from typing import Optional, Any
from logging import Logger
from pathlib import Path
from rich.progress import Progress
from pipeline.pipeline_context import PipelineContext
from util.device_utils import clean_device_cache

class PipelineStageConfiguration:
    output: Path
    temp: Path

    def __init__(
            self,
            name: str, 
            output_root: Path, 
            temp: Path, 
            device: torch.device, 
            torch_dtype: Any,
            log: Logger
            ):
        self.name = name
        if output_root is not None:
            self.output = output_root / name
            self.output.mkdir(parents=True, exist_ok=True)
        else:
            self.output = None

        if temp is not None:
            self.temp = temp / name
            self.temp.mkdir(parents=True, exist_ok=True)    
        else:
            self.temp = None

        self.device = device
        self.torch_dtype = torch_dtype
        self.log = log

class PipelineStage:
    def __init__(self, config: PipelineStageConfiguration) -> None:
        self.name = config.name
        self.config = config
        self.device = config.device
        self.torch_dtype = config.torch_dtype
        self.total_tasks = None

    def model_names(self) -> list[str]:
        return []

    def _log_memory_usage(self, value):
        mb = value / 1024 / 1024

        if mb < 2048:  # less than 2GB
            formatted = f"{mb:.0f} MB"
        else:
            gb = mb / 1024.0
            formatted = f"{gb:.1f} GB"


        BOLD = "\033[1m"
        BLUE = "\033[94m"
        RESET = "\033[0m"
        
        print(f"{BLUE}Peak Memory({self.name}): {BOLD}{formatted}{RESET}")

    def log_memory_usage(self):
        if self.device.type == "cuda":
            self._log_memory_usage(torch.cuda.max_memory_allocated())
        elif self.device.type == "mps":
            self._log_memory_usage(torch.mps.driver_allocated_memory())

    def clean_up(self):
        clean_device_cache(self.device)

    def log_info(self, message):
        self.config.log.info(message)

    def log_warning(self, message):
        self.config.log.warning(message)

    def log_error(self, message):
        self.config.log.error(message)

    def run(self, context: PipelineContext) -> PipelineContext:
        return context

    def set_total_tasks(self, count: int):
        self.total_tasks = count

    def create_progress(self, count: int, label: Optional[str] = None):
        if label is None:
            label = self.name

        return self.progress.add_task("  " + label, total=count)

    def advance_progress(self, sub_task):
        self.progress.advance(sub_task)
        
        task = next(t for t in self.progress.tasks if t.id == sub_task)
        sub_total = task.total if task.total is not None else task.completed + 1
        total_tasks = self.total_tasks if self.total_tasks is not None else 1
        self.progress.advance(self.main_task, 1 / (sub_total * total_tasks))

    def finish_progress(self, task):
        # Snap main task forward by whatever fraction this stage didn't account for
        t = next(t for t in self.progress.tasks if t.id == task)
        sub_total = t.total if t.total is not None else 1
        total_tasks = self.total_tasks if self.total_tasks is not None else 1
        remaining = (sub_total - t.completed) / (sub_total * total_tasks)
        if remaining > 0:
            self.progress.advance(self.main_task, remaining)
        self.progress.remove_task(task)

    def _set_progress(self, progress: Progress, main_task):
        self.progress = progress
        self.main_task = main_task

    
    