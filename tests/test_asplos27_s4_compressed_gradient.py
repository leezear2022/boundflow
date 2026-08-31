"""S4-1C TIR compressed-gradient and terminal-copy gates."""

# mypy: disable-error-code=import-untyped
# pylint: disable=missing-function-docstring,too-many-locals,not-callable
# pylint: disable=import-outside-toplevel

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.backends.tvm.asplos27_s4_compressed_gradient import (
    S4_COMPRESSED_GRADIENT_QNAN_BITS_V1,
    S4_DBETA31_SYMBOL_V1,
    S4_GRADIENT_EXPORTED_SYMBOLS_V1,
    S4_GRADIENT_SITE_SPECS_V1,
    build_s4_compressed_gradient_tir_modules_v1,
    compile_s4_compressed_gradient_v1,
)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("S4-1C CUDA fixture is unavailable")
    pytest.importorskip("tvm")
    return torch.device("cuda", torch.cuda.current_device())


def _view(tensor: torch.Tensor):
    import tvm

    return tvm.runtime.from_dlpack(tensor)


def test_s4_gradient_module_has_exact_thirteen_symbol_inventory() -> None:
    pytest.importorskip("tvm")
    unscheduled, scheduled = build_s4_compressed_gradient_tir_modules_v1()
    assert len(unscheduled.functions) == len(scheduled.functions) == 13
    script = scheduled.script()
    assert (
        tuple(symbol for symbol in S4_GRADIENT_EXPORTED_SYMBOLS_V1 if symbol in script)
        == S4_GRADIENT_EXPORTED_SYMBOLS_V1
    )
    assert "0x7FC00000" in script or "2143289344" in script
    assert "threadIdx.x" in script and "blockIdx.x" in script


def test_s4_all_six_dalpha_emitters_match_independent_formula() -> None:
    device = _device()
    torch.manual_seed(4101)
    compiled = compile_s4_compressed_gradient_v1()
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        for spec in S4_GRADIENT_SITE_SPECS_V1:
            coefficient = torch.randn(
                (6, 1, spec.feature_count), device=device, dtype=torch.float32
            ).contiguous()
            adjoint = torch.randn_like(coefficient).contiguous()
            center = torch.randn((6, spec.feature_count), device=device) * 0.1
            radius = torch.rand_like(center) + 0.05
            lower = (center - radius).contiguous()
            upper = (center + radius).contiguous()
            alpha = torch.rand((6, spec.alpha_width), device=device).contiguous()
            indices = torch.linspace(
                0,
                spec.feature_count - 1,
                spec.alpha_width,
                device=device,
            ).to(torch.int32)
            indices = torch.unique_consecutive(indices)
            if indices.numel() != spec.alpha_width:
                indices = torch.arange(
                    spec.alpha_width, device=device, dtype=torch.int32
                )
            upstream = torch.randn((6, 1), device=device).contiguous()
            output = torch.empty_like(alpha)
            compiled.executable[spec.dalpha_symbol](
                *map(
                    _view,
                    (
                        coefficient,
                        adjoint,
                        lower,
                        upper,
                        alpha,
                        indices,
                        upstream,
                        output,
                    ),
                )
            )
            selected_a = coefficient[:, 0].gather(
                1, indices.to(torch.int64).view(1, -1).expand(6, -1)
            )
            selected_v = adjoint[:, 0].gather(
                1, indices.to(torch.int64).view(1, -1).expand(6, -1)
            )
            selected_l = lower.gather(
                1, indices.to(torch.int64).view(1, -1).expand(6, -1)
            )
            selected_u = upper.gather(
                1, indices.to(torch.int64).view(1, -1).expand(6, -1)
            )
            active = (selected_l < 0) & (selected_u > 0) & (selected_a >= 0)
            reference = torch.where(
                active, upstream * selected_a * selected_v, torch.zeros_like(alpha)
            )
            torch.testing.assert_close(output, reference, rtol=1e-6, atol=1e-6)
    stream.synchronize()


def test_s4_gradient_poison_safe_index_beta_and_terminal_copy() -> None:
    device = _device()
    compiled = compile_s4_compressed_gradient_v1()
    spec = S4_GRADIENT_SITE_SPECS_V1[-1]
    coefficient = torch.ones((6, 1, 100), device=device)
    adjoint = torch.arange(600, device=device, dtype=torch.float32).view(6, 1, 100)
    lower = torch.full((6, 100), -1.0, device=device)
    upper = torch.full((6, 100), 1.0, device=device)
    alpha = torch.full((6, 27), 0.5, device=device)
    indices = torch.arange(27, device=device, dtype=torch.int32)
    upstream = torch.full((6, 1), -1.0, device=device)
    dalpha = torch.empty((6, 27), device=device)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        indices[-1] = 1000
        compiled.executable[spec.dalpha_symbol](
            *map(
                _view,
                (coefficient, adjoint, lower, upper, alpha, indices, upstream, dalpha),
            )
        )
        location = torch.tensor(
            [17, 17, 31, 17, 17, 31], device=device, dtype=torch.int32
        ).view(6, 1)
        sign = torch.tensor(
            [1, 1, 1, -1, -1, -1], device=device, dtype=torch.int8
        ).view(6, 1)
        dbeta = torch.empty((6, 1), device=device)
        compiled.executable[S4_DBETA31_SYMBOL_V1](
            *map(_view, (adjoint, location, sign, upstream, dbeta))
        )
    stream.synchronize()
    assert (
        dalpha[-1, -1].view(torch.uint32).item() == S4_COMPRESSED_GRADIENT_QNAN_BITS_V1
    )
    expected_beta = torch.tensor(
        [17.0, 117.0, 231.0, -317.0, -417.0, -531.0], device=device
    ).view(6, 1)
    torch.testing.assert_close(dbeta, expected_beta)

    target = torch.empty_like(coefficient)
    with torch.cuda.stream(stream):
        compiled.executable[spec.copy_symbol](_view(coefficient), _view(target))
    stream.synchronize()
    assert torch.equal(target, coefficient)


def test_s4_gradient_invalid_inputs_poison_instead_of_silent_zero() -> None:
    device = _device()
    compiled = compile_s4_compressed_gradient_v1()
    spec = S4_GRADIENT_SITE_SPECS_V1[-1]
    base = {
        "coefficient": torch.ones((6, 1, 100), device=device),
        "adjoint": torch.ones((6, 1, 100), device=device),
        "lower": torch.full((6, 100), -1.0, device=device),
        "upper": torch.full((6, 100), 1.0, device=device),
        "alpha": torch.full((6, 27), 0.5, device=device),
        "indices": torch.arange(27, device=device, dtype=torch.int32),
        "upstream": torch.full((6, 1), -1.0, device=device),
    }

    def run(changes: dict[str, float]) -> int:
        values = {name: tensor.clone() for name, tensor in base.items()}
        for name, replacement in changes.items():
            values[name][0, 0] = replacement
        output = torch.empty((6, 27), device=device)
        compiled.executable[spec.dalpha_symbol](
            *map(
                _view,
                (
                    values["coefficient"],
                    values["adjoint"],
                    values["lower"],
                    values["upper"],
                    values["alpha"],
                    values["indices"],
                    values["upstream"],
                    output,
                ),
            )
        )
        return int(output[0, 0].view(torch.uint32).item())

    invalid = (
        {"coefficient": float("nan")},
        {"adjoint": float("inf")},
        {"lower": float("nan")},
        {"upper": float("inf")},
        {"alpha": float("nan")},
        {"upstream": float("inf")},
        {"lower": 2.0, "upper": 1.0},
        {"alpha": -0.01},
        {"alpha": 1.01},
    )
    assert all(
        run(changes) == S4_COMPRESSED_GRADIENT_QNAN_BITS_V1 for changes in invalid
    )
    assert run({"coefficient": -1.0}) == 0
    assert run({"lower": 0.1, "upper": 1.0}) == 0
    assert run({"alpha": 0.0}) != S4_COMPRESSED_GRADIENT_QNAN_BITS_V1
    assert run({"alpha": 1.0}) != S4_COMPRESSED_GRADIENT_QNAN_BITS_V1

    adjoint = torch.ones((6, 1, 100), device=device)
    location = torch.full((6, 1), 17, dtype=torch.int32, device=device)
    sign = torch.ones((6, 1), dtype=torch.int8, device=device)
    upstream = torch.full((6, 1), -1.0, device=device)
    output = torch.empty((6, 1), device=device)
    for mutation in ("location", "sign", "adjoint", "upstream"):
        current_location = location.clone()
        current_sign = sign.clone()
        current_adjoint = adjoint.clone()
        current_upstream = upstream.clone()
        if mutation == "location":
            current_location[0, 0] = 1000
        elif mutation == "sign":
            current_sign[0, 0] = 0
        elif mutation == "adjoint":
            current_adjoint[0, 0, 17] = float("inf")
        else:
            current_upstream[0, 0] = float("nan")
        compiled.executable[S4_DBETA31_SYMBOL_V1](
            *map(
                _view,
                (
                    current_adjoint,
                    current_location,
                    current_sign,
                    current_upstream,
                    output,
                ),
            )
        )
        assert (
            output[0, 0].view(torch.uint32).item()
            == S4_COMPRESSED_GRADIENT_QNAN_BITS_V1
        )


def test_s4_gradient_compiled_identity_is_fail_closed() -> None:
    _device()
    compiled = compile_s4_compressed_gradient_v1()
    compiled.validate()
    mutations = (
        replace(compiled, unscheduled_tir_json=compiled.unscheduled_tir_json + " "),
        replace(compiled, scheduled_tir_json=compiled.scheduled_tir_json + " "),
        replace(compiled, device_source=compiled.device_source + " "),
        replace(compiled, exported_symbols=compiled.exported_symbols[:-1]),
        replace(compiled, global_workspace_bytes=4),
        replace(compiled, performance_claimed=True),
    )
    for changed in mutations:
        with pytest.raises(ValueError, match="compiled identity differs"):
            changed.validate()
