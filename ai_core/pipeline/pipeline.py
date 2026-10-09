"""
Data Pipeline — composable, typed ETL steps for AI workflows.

Usage:
    pipeline = (
        Pipeline("ingest-docs")
        .step("load",    load_files,      description="Load raw files")
        .step("clean",   clean_text,      description="Strip noise")
        .step("embed",   embed_chunks,    description="Generate embeddings")
        .step("store",   upsert_to_db,   description="Upsert into vector store")
    )
    result = await pipeline.run({"path": "./docs"})
    print(result.outputs)     # dict of step_name → output
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ai_core.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StepResult:
    name: str
    success: bool
    output: Any
    elapsed_s: float
    error: Optional[str] = None


@dataclass
class PipelineResult:
    name: str
    steps: List[StepResult] = field(default_factory=list)
    success: bool = True
    total_elapsed_s: float = 0.0

    @property
    def outputs(self) -> Dict[str, Any]:
        return {s.name: s.output for s in self.steps if s.success}

    def summary(self) -> str:
        lines = [f"Pipeline '{self.name}' — {'✓' if self.success else '✗'} ({self.total_elapsed_s:.2f}s)"]
        for s in self.steps:
            icon = "✓" if s.success else "✗"
            lines.append(f"  {icon} {s.name:<25} {s.elapsed_s:.2f}s" +
                         (f"  ERROR: {s.error}" if s.error else ""))
        return "\n".join(lines)


@dataclass
class Step:
    name: str
    fn: Callable
    description: str = ""
    skip_on_error: bool = False     # continue pipeline even if this step fails
    parallel_group: Optional[str] = None  # steps in the same group run concurrently

    async def execute(self, data: Any) -> Any:
        if asyncio.iscoroutinefunction(self.fn):
            return await self.fn(data)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fn, data)


class Pipeline:
    """
    Sequential (and optionally parallel) data processing pipeline.
    Each step receives the output of the previous step as its input.
    """

    def __init__(self, name: str, on_error: str = "stop"):
        """
        Args:
            name: Pipeline display name.
            on_error: "stop" (default) aborts on first failure, "continue" logs and moves on.
        """
        self.name = name
        self.on_error = on_error
        self._steps: List[Step] = []

    def step(
        self,
        name: str,
        fn: Callable,
        *,
        description: str = "",
        skip_on_error: bool = False,
        parallel_group: Optional[str] = None,
    ) -> "Pipeline":
        """Add a step. Returns self for chaining."""
        self._steps.append(Step(
            name=name, fn=fn, description=description,
            skip_on_error=skip_on_error, parallel_group=parallel_group,
        ))
        return self

    async def run(self, initial_data: Any = None) -> PipelineResult:
        """Execute all steps sequentially, passing output of each as input to next."""
        start_total = time.perf_counter()
        result = PipelineResult(name=self.name)
        data = initial_data

        logger.info("pipeline.start", name=self.name, steps=len(self._steps))

        for step in self._steps:
            t0 = time.perf_counter()
            try:
                output = await step.execute(data)
                elapsed = time.perf_counter() - t0
                step_result = StepResult(name=step.name, success=True, output=output, elapsed_s=elapsed)
                data = output     # pass to next step
                logger.debug("pipeline.step_ok", step=step.name, elapsed_s=round(elapsed, 3))
            except Exception as e:
                elapsed = time.perf_counter() - t0
                step_result = StepResult(name=step.name, success=False, output=None,
                                         elapsed_s=elapsed, error=str(e))
                logger.error("pipeline.step_fail", step=step.name, error=str(e))
                result.steps.append(step_result)
                result.success = False
                if self.on_error == "stop" and not step.skip_on_error:
                    break
                # continue with whatever data we had
            else:
                result.steps.append(step_result)

        result.total_elapsed_s = time.perf_counter() - start_total
        logger.info("pipeline.done", name=self.name, success=result.success,
                    elapsed_s=round(result.total_elapsed_s, 3))
        return result

    def __repr__(self) -> str:
        return f"Pipeline(name={self.name!r}, steps={[s.name for s in self._steps]})"
