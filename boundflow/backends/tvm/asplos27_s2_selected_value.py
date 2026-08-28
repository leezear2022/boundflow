"""S2 selected-value CROWN graph lowered through Relax, TIR, and cuDNN."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-locals
# pylint: disable=too-many-statements,too-many-arguments,too-many-positional-arguments
# pylint: disable=missing-function-docstring,invalid-name,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any

S2_SELECTED_VALUE_SCHEMA = "boundflow.asplos27-s2-selected-value/v1"
S2_SELECTED_VALUE_FUNCTION = "boundflow_s2_selected_value"
S2_SELECTED_VALUE_CUDNN_FUNCTIONS = 4
S2_SELECTED_VALUE_CUDNN_CALLS = 5


@dataclass(frozen=True)
class CompiledS2SelectedValueV1:
    """Compiled Relax executable and identities for the exact P-anchor chain."""

    executable: Any
    function_name: str
    source_relax_ir_hash: str
    partitioned_relax_ir_hash: str
    lowered_relax_ir_hash: str
    device_source_hashes: tuple[str, ...]
    target: str
    cudnn_partition_function_count: int
    cudnn_conv_call_count: int
    selected_tir_count: int
    compile_ms: float

    def validate(self) -> None:
        hashes = (
            self.source_relax_ir_hash,
            self.partitioned_relax_ir_hash,
            self.lowered_relax_ir_hash,
            *self.device_source_hashes,
        )
        if (
            self.function_name != S2_SELECTED_VALUE_FUNCTION
            or any(len(value) != 64 for value in hashes)
            or not self.device_source_hashes
            or not self.target
            or self.cudnn_partition_function_count != S2_SELECTED_VALUE_CUDNN_FUNCTIONS
            or self.cudnn_conv_call_count != S2_SELECTED_VALUE_CUDNN_CALLS
            or self.selected_tir_count != 4
            or self.compile_ms <= 0.0
        ):
            raise ValueError("S2 selected-value compiled identity differs")


def _schedule_elementwise(tvm: Any, symbol: str, primfunc: Any, block_name: str):
    schedule = tvm.tir.Schedule(tvm.IRModule({symbol: primfunc}))
    block = schedule.get_block(block_name, func_name=symbol)
    loops = schedule.get_loops(block)
    fused = schedule.fuse(*loops) if len(loops) > 1 else loops[0]
    outer, inner = schedule.split(fused, factors=[None, 256])
    schedule.bind(outer, "blockIdx.x")
    schedule.bind(inner, "threadIdx.x")
    return schedule.mod[symbol]


def _input_select_primfunc():
    import tvm
    from tvm import te

    lower = te.placeholder((6, 3, 32, 32), "float32", name="input_lower")
    upper = te.placeholder((6, 3, 32, 32), "float32", name="input_upper")
    sign = te.placeholder((18432,), "int8", name="input_sign")
    output = te.compute(
        (6, 3, 32, 32),
        lambda d, c, h, w: tvm.tir.if_then_else(
            sign[d * 3072 + c * 1024 + h * 32 + w] != 0,
            lower[d, c, h, w],
            upper[d, c, h, w],
        ),
        name="selected_input",
    )
    return te.create_prim_func([lower, upper, sign, output]).with_attr(
        "boundflow.schema_version", S2_SELECTED_VALUE_SCHEMA
    )


def _selected_relu_primfunc(
    *, channels: int, height: int, width: int, alpha_width: int
):
    import tvm
    from tvm import te

    pre = te.placeholder((6, channels, height, width), "float32", name="pre")
    sign = te.placeholder(
        (6 * channels * height * width,), "int8", name="incoming_sign"
    )
    lower = te.placeholder((6, channels, height, width), "float32", name="lower")
    upper = te.placeholder((6, channels, height, width), "float32", name="upper")
    alpha = te.placeholder((2, 1, 6, alpha_width), "float32", name="alpha")
    alpha_map = te.placeholder((channels * height * width,), "int32", name="alpha_map")
    zero = tvm.tir.const(0.0, "float32")
    one = tvm.tir.const(1.0, "float32")
    epsilon = tvm.tir.const(1.1920928955078125e-07, "float32")

    def selected(d, c, h, w):  # type: ignore[no-untyped-def]
        feature = c * height * width + h * width + w
        flat = d * channels * height * width + feature
        lookup = alpha_map[feature]
        compact = alpha[0, 0, d, tvm.tir.max(lookup, 0)]
        lower_alpha = tvm.tir.if_then_else(
            lookup >= 0, tvm.tir.min(tvm.tir.max(compact, zero), one), zero
        )
        ambiguous = tvm.tir.all(lower[d, c, h, w] < zero, upper[d, c, h, w] > zero)
        lower_slope = tvm.tir.if_then_else(
            ambiguous,
            lower_alpha,
            tvm.tir.if_then_else(lower[d, c, h, w] >= zero, one, zero),
        )
        upper_slope = tvm.tir.if_then_else(
            lower[d, c, h, w] >= zero,
            one,
            tvm.tir.if_then_else(
                upper[d, c, h, w] <= zero,
                zero,
                upper[d, c, h, w]
                / tvm.tir.max(upper[d, c, h, w] - lower[d, c, h, w], epsilon),
            ),
        )
        slope = tvm.tir.if_then_else(sign[flat] != 0, lower_slope, upper_slope)
        intercept = tvm.tir.if_then_else(
            tvm.tir.all(sign[flat] == 0, ambiguous),
            -lower[d, c, h, w] * upper_slope,
            zero,
        )
        return pre[d, c, h, w] * slope + intercept

    output = te.compute((6, channels, height, width), selected, name="selected_relu")
    return te.create_prim_func(
        [pre, sign, lower, upper, alpha, alpha_map, output]
    ).with_attr("boundflow.schema_version", S2_SELECTED_VALUE_SCHEMA)


def _tensor_var(relax: Any, name: str, shape: tuple[int, ...], dtype: str = "float32"):
    return relax.Var(name, relax.TensorStructInfo(shape, dtype))


def build_s2_selected_value_relax_module_v1():
    """Build the exact five-convolution selected-value graph for P-anchor."""

    import tvm
    from tvm import relax

    builder = relax.BlockBuilder()
    parameters = [
        _tensor_var(relax, "input_lower", (6, 3, 32, 32)),
        _tensor_var(relax, "input_upper", (6, 3, 32, 32)),
        _tensor_var(relax, "sign_ainput", (18432,), "int8"),
        _tensor_var(relax, "weight0", (8, 3, 3, 3)),
        _tensor_var(relax, "bias0", (8,)),
        _tensor_var(relax, "lower17", (6, 8, 16, 16)),
        _tensor_var(relax, "upper17", (6, 8, 16, 16)),
        _tensor_var(relax, "alpha17", (2, 1, 6, 164)),
        _tensor_var(relax, "alpha_map17", (2048,), "int32"),
        _tensor_var(relax, "sign_a18", (12288,), "int8"),
        _tensor_var(relax, "weight2", (16, 8, 3, 3)),
        _tensor_var(relax, "bias2", (16,)),
        _tensor_var(relax, "lower19", (6, 16, 8, 8)),
        _tensor_var(relax, "upper19", (6, 16, 8, 8)),
        _tensor_var(relax, "alpha19", (2, 1, 6, 132)),
        _tensor_var(relax, "alpha_map19", (1024,), "int32"),
        _tensor_var(relax, "sign_a20", (6144,), "int8"),
        _tensor_var(relax, "weight4", (16, 16, 3, 3)),
        _tensor_var(relax, "bias4", (16,)),
        _tensor_var(relax, "weight5", (16, 8, 1, 1)),
        _tensor_var(relax, "bias5", (16,)),
        _tensor_var(relax, "lower23", (6, 16, 8, 8)),
        _tensor_var(relax, "upper23", (6, 16, 8, 8)),
        _tensor_var(relax, "alpha23", (2, 1, 6, 121)),
        _tensor_var(relax, "alpha_map23", (1024,), "int32"),
        _tensor_var(relax, "sign_a24", (6144,), "int8"),
        _tensor_var(relax, "weight8", (16, 16, 3, 3)),
        _tensor_var(relax, "bias8", (16,)),
    ]

    input_name = "boundflow_s2_select_input_tir"
    relu17_name = "boundflow_s2_select_relu17_tir"
    relu19_name = "boundflow_s2_select_relu19_tir"
    relu23_name = "boundflow_s2_select_relu23_tir"
    input_global = builder.add_func(
        _schedule_elementwise(
            tvm, input_name, _input_select_primfunc(), "selected_input"
        ).with_attr("global_symbol", input_name),
        input_name,
    )
    relu17_global = builder.add_func(
        _schedule_elementwise(
            tvm,
            relu17_name,
            _selected_relu_primfunc(channels=8, height=16, width=16, alpha_width=164),
            "selected_relu",
        ).with_attr("global_symbol", relu17_name),
        relu17_name,
    )
    relu19_global = builder.add_func(
        _schedule_elementwise(
            tvm,
            relu19_name,
            _selected_relu_primfunc(channels=16, height=8, width=8, alpha_width=132),
            "selected_relu",
        ).with_attr("global_symbol", relu19_name),
        relu19_name,
    )
    relu23_global = builder.add_func(
        _schedule_elementwise(
            tvm,
            relu23_name,
            _selected_relu_primfunc(channels=16, height=8, width=8, alpha_width=121),
            "selected_relu",
        ).with_attr("global_symbol", relu23_name),
        relu23_name,
    )

    def conv_bias(  # type: ignore[no-untyped-def]
        data, weight, bias, *, strides, padding, channels
    ):
        conv = builder.emit(
            relax.op.nn.conv2d(
                data,
                weight,
                strides=strides,
                padding=padding,
                dilation=(1, 1),
                groups=1,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_dtype="float32",
            )
        )
        shaped_bias = builder.emit(relax.op.reshape(bias, (1, channels, 1, 1)))
        return builder.emit(relax.op.add(conv, shaped_bias))

    with builder.function(S2_SELECTED_VALUE_FUNCTION, parameters):
        with builder.dataflow():
            selected_input = builder.emit(
                relax.call_tir(
                    input_global,
                    relax.Tuple(parameters[0:3]),
                    out_sinfo=relax.TensorStructInfo((6, 3, 32, 32), "float32"),
                )
            )
            pre17 = conv_bias(
                selected_input,
                parameters[3],
                parameters[4],
                strides=(2, 2),
                padding=(1, 1),
                channels=8,
            )
            selected17 = builder.emit(
                relax.call_tir(
                    relu17_global,
                    relax.Tuple([pre17, *parameters[9:10], *parameters[5:9]]),
                    out_sinfo=relax.TensorStructInfo((6, 8, 16, 16), "float32"),
                )
            )
            pre19 = conv_bias(
                selected17,
                parameters[10],
                parameters[11],
                strides=(2, 2),
                padding=(1, 1),
                channels=16,
            )
            selected19 = builder.emit(
                relax.call_tir(
                    relu19_global,
                    relax.Tuple([pre19, *parameters[16:17], *parameters[12:16]]),
                    out_sinfo=relax.TensorStructInfo((6, 16, 8, 8), "float32"),
                )
            )
            main = conv_bias(
                selected19,
                parameters[17],
                parameters[18],
                strides=(1, 1),
                padding=(1, 1),
                channels=16,
            )
            shortcut = conv_bias(
                selected17,
                parameters[19],
                parameters[20],
                strides=(2, 2),
                padding=(0, 0),
                channels=16,
            )
            pre23 = builder.emit(relax.op.add(main, shortcut))
            selected23 = builder.emit(
                relax.call_tir(
                    relu23_global,
                    relax.Tuple([pre23, *parameters[25:26], *parameters[21:25]]),
                    out_sinfo=relax.TensorStructInfo((6, 16, 8, 8), "float32"),
                )
            )
            pre25 = conv_bias(
                selected23,
                parameters[26],
                parameters[27],
                strides=(1, 1),
                padding=(1, 1),
                channels=16,
            )
            output = builder.emit_output(pre25)
        builder.emit_func_output(output)
    return builder.get()


def _module_sources(executable: Any) -> tuple[str, ...]:
    sources: list[str] = []

    def visit(module: Any) -> None:
        try:
            source = module.inspect_source()
        except (AttributeError, RuntimeError, TypeError):
            source = ""
        if source:
            sources.append(source)
        for imported in getattr(module, "imports", ()):
            visit(imported)

    visit(executable.mod)
    return tuple(sources)


def compile_s2_selected_value_v1(*, device_index: int = 0) -> CompiledS2SelectedValueV1:
    """Compile the P-anchor selected-value graph; no runtime fallback is admitted."""

    import torch
    import tvm
    from tvm import dlight as dl
    from tvm import relax, transform
    from tvm.relax.backend.cuda.cudnn import partition_for_cudnn

    source = build_s2_selected_value_relax_module_v1()
    source_hash = hashlib.sha256(tvm.ir.save_json(source).encode()).hexdigest()
    partitioned = partition_for_cudnn(source)
    partition_function_count = sum(
        1
        for function in partitioned.functions.values()
        if function.attrs is not None
        and str(function.attrs.get("Codegen", "")) == "cudnn"
    )
    cudnn_globals = {
        global_var
        for global_var, function in partitioned.functions.items()
        if function.attrs is not None
        and str(function.attrs.get("Codegen", "")) == "cudnn"
    }
    cudnn_call_count = 0

    def count_cudnn_calls(expression: Any) -> None:
        nonlocal cudnn_call_count
        if isinstance(expression, relax.Call) and expression.op in cudnn_globals:
            cudnn_call_count += 1

    relax.analysis.post_order_visit(
        partitioned[S2_SELECTED_VALUE_FUNCTION].body, count_cudnn_calls
    )
    if (
        partition_function_count != S2_SELECTED_VALUE_CUDNN_FUNCTIONS
        or cudnn_call_count != S2_SELECTED_VALUE_CUDNN_CALLS
    ):
        raise RuntimeError(
            "S2 selected-value cuDNN partition/call count differs: "
            f"{partition_function_count}/{cudnn_call_count}"
        )
    partitioned_hash = hashlib.sha256(
        tvm.ir.save_json(partitioned).encode()
    ).hexdigest()
    lowered = relax.transform.RunCodegen()(partitioned)
    lowered_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode()).hexdigest()
    major, minor = torch.cuda.get_device_capability(device_index)
    target = tvm.target.Target(f"cuda -arch=sm_{major}{minor}", host="llvm")
    default_schedule = dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),
        dl.gpu.GEMV(),
        dl.gpu.Reduction(),
        dl.gpu.GeneralReduction(),
        dl.gpu.Fallback(),
    )
    pipeline = transform.Sequential(
        [default_schedule, tvm.tir.get_default_tir_pipeline(target)]
    )
    started = time.perf_counter()
    executable = relax.build(lowered, target=target, tir_pipeline=pipeline)
    compile_ms = (time.perf_counter() - started) * 1000.0
    sources = _module_sources(executable)
    if not sources:
        sources = (tvm.ir.save_json(lowered),)
    compiled = CompiledS2SelectedValueV1(
        executable=executable,
        function_name=S2_SELECTED_VALUE_FUNCTION,
        source_relax_ir_hash=source_hash,
        partitioned_relax_ir_hash=partitioned_hash,
        lowered_relax_ir_hash=lowered_hash,
        device_source_hashes=tuple(
            hashlib.sha256(value.encode()).hexdigest() for value in sources
        ),
        target=str(target),
        cudnn_partition_function_count=partition_function_count,
        cudnn_conv_call_count=cudnn_call_count,
        selected_tir_count=4,
        compile_ms=compile_ms,
    )
    compiled.validate()
    return compiled


__all__ = [
    "CompiledS2SelectedValueV1",
    "S2_SELECTED_VALUE_CUDNN_CALLS",
    "S2_SELECTED_VALUE_CUDNN_FUNCTIONS",
    "S2_SELECTED_VALUE_FUNCTION",
    "build_s2_selected_value_relax_module_v1",
    "compile_s2_selected_value_v1",
]
