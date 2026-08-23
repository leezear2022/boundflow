"""Contract tests for prepared CIBC IBP CUDA graph plans."""

# pylint: disable=missing-function-docstring

from boundflow.runtime.cibc_ibp_graph import CIBCIBPCUDAGraphPlanV1


def test_cibc_ibp_cuda_graph_plan_is_exported() -> None:
    assert CIBCIBPCUDAGraphPlanV1.__name__ == "CIBCIBPCUDAGraphPlanV1"
