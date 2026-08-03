"""Parser tests for PR-12 post-codegen evidence."""

from scripts.profile_phase7a_pr12_codegen import parse_cuda_kernel_names, parse_ptxas


def test_parse_ptxas_extracts_spill_and_register_metrics() -> None:
    stderr = """ptxas info    : Function properties for main_kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 32 registers, used 0 barriers, 392 bytes cmem[0]
"""

    assert parse_ptxas(stderr) == {
        "main_kernel": {
            "stack_frame_bytes": 0,
            "spill_store_bytes": 0,
            "spill_load_bytes": 0,
            "registers_per_thread": 32,
        }
    }


def test_parse_cuda_kernel_names_ignores_forward_declaration() -> None:
    source = """extern "C" __global__ void __launch_bounds__(128) main_kernel(float* x);
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* x) { x[0] = 0; }
"""

    assert parse_cuda_kernel_names(source) == ["main_kernel"]
