from __future__ import annotations

from typing import Optional, Protocol

from ..ir.primal import BFPrimalProgram
from ..planner.interval_v0 import plan_interval_ibp_v0
from .task_executor import LinfInputSpec, PythonTaskExecutor
from ..domains.interval import IntervalState


class Executor(Protocol):
    def run_ibp(
        self, program: BFPrimalProgram, input_spec: LinfInputSpec, *, output_value: Optional[str] = None
    ) -> IntervalState: ...


class PythonInterpreter:
    """
    兼容层：保留 Phase 3 的 API（输入 BFPrimalProgram），但底层走 Phase 4 的 Task 路径。
    """

    def __init__(self):
        self._task_executor = PythonTaskExecutor()

    def run_ibp(
        self, program: BFPrimalProgram, input_spec: LinfInputSpec, *, output_value: Optional[str] = None
    ) -> IntervalState:
        task_module = plan_interval_ibp_v0(program)
        return self._task_executor.run_ibp(task_module, input_spec, output_value=output_value)

