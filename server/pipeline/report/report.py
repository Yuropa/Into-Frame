from datetime import date
from pathlib import Path
from typing import Optional

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey


class ReportGenerationStage(PipelineStage):
    """
    Final pipeline stage: collects ReportSections contributed by upstream stages
    and renders a LaTeX-style PDF report to the output directory.

    No ML models are loaded. The stage reads context keys:
      ContextKey.INPUT         — input photograph (cover page)
      ContextKey.INPUT_CAPTION — scene description (abstract)

    All ReportSections must have been added to the context via
    context.add_report_section() before this stage runs.
    """

    def run(self, context: PipelineContext) -> PipelineContext:
        from pipeline.report import pdf_generator

        task = self.create_progress(2, "Generating report…")

        sections = context.report_sections()
        input_img = context.image(ContextKey.INPUT)
        caption = context.object(ContextKey.INPUT_CAPTION)

        pil_input = input_img.image if input_img is not None else None

        self.advance_progress(task)

        if self.output is not None:
            out_path = self.output / "report.pdf"
            pdf_generator.generate(
                output_path=out_path,
                sections=sections,
                input_image=pil_input,
                caption=caption,
                run_date=date.today().isoformat(),
            )
            # Also surface it one level up for easy access
            top_level = self.output.parent / "report.pdf"
            import shutil
            shutil.copy2(out_path, top_level)
            self.log_info(f"Report written to {top_level}")
        else:
            self.log_warning("No output path set — report not written")

        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        if self.output is None:
            return False
        return (self.output / "report.pdf").exists()

    def contribute_report(self, context: PipelineContext) -> None:
        return None

    def model_names(self) -> list[str]:
        return []
