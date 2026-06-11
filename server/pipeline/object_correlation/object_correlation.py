from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_correlation.object_correlation_result import ObjectCorrelationResult, ObjectGroupStats


class ObjectCorrelationStage(PipelineStage):
    """
    Groups detected objects by their type label to enable cross-object analysis.

    Reads:  ContextKey.OBJECT_COUNT, metadata_{i} (with 'type' field from ObjectTypingStage)
    Writes: ContextKey.OBJECT_CORRELATION (ObjectCorrelationResult)
    """

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            self.log_info("No objects to correlate, skipping")
            return context

        result = ObjectCorrelationResult()
        correlation_task = self.create_progress(object_count, "Correlating objects...")

        for idx in range(object_count):
            metadata = context.input_object(f"metadata_{idx}") or {}
            obj_type = metadata.get("type") or "unknown"

            if obj_type not in result.groups:
                result.groups[obj_type] = ObjectGroupStats(object_type=obj_type)
            result.groups[obj_type].indices.append(idx)

            self.advance_progress(correlation_task)

        for obj_type, stats in result.groups.items():
            self.log_info(f"  {obj_type}: {stats.count} object(s) at indices {stats.indices}")

        context.add_object(ContextKey.OBJECT_CORRELATION, result)
        self.finish_progress(correlation_task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(ContextKey.OBJECT_CORRELATION) is not None
