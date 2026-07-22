import queue
from pathlib import Path
from typing import List, Optional, Tuple
from pipeline.pipeline import Pipeline, PipelineContext
from pipeline.pipeline_input import PipelineInputItem, PipelineInput

class PipelineRunner:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        # (item, context_dir) for every sample processed by the most recent run() call.
        self.processed: List[Tuple[PipelineInputItem, Optional[Path]]] = []

    def run(self, input: PipelineInput, progress_queue: Optional[queue.SimpleQueue] = None) -> Optional[PipelineContext]:
        total = input.count()
        last_context = None
        self.processed = []

        for i, item in enumerate(input.all_images()):
            if progress_queue is not None:
                inner_queue = queue.SimpleQueue()

                def forward_progress():
                    while not inner_queue.empty():
                        inner = inner_queue.get()
                        progress_queue.put({
                            **inner,
                            "current": i + 1,
                            "total": total,
                            "progress": (i + inner.get("progress", 1.0)) / total,
                        })

                last_context = self.pipeline.run(item, inner_queue)
                forward_progress()
            else:
                last_context = self.pipeline.run(item, None)

            self.processed.append((item, self.pipeline.context_path()))

        return last_context
