import queue
from typing import Optional
from pipeline.pipeline import Pipeline, PipelineContext
from pipeline.pipeline_input import PipelineInputItem, PipelineInput

class PipelineRunner:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def run(self, input: PipelineInput, progress_queue: Optional[queue.SimpleQueue] = None) -> Optional[PipelineContext]:
        total = input.count()
        last_context = None

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

        return last_context
