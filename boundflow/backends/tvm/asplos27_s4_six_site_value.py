"""S4-1B six-site selected-value graph lowered through Relax/TIR/cuDNN.

The module owns the exact 42-read/7-write production ABI.  It deliberately
contains no coefficient recomputation, optimizer mutation, timing, fallback,
or performance claim.  All six persistent value slots are caller-owned views.
"""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-locals
# pylint: disable=too-many-statements,too-many-arguments,too-many-positional-arguments
# pylint: disable=missing-function-docstring,invalid-name,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,too-many-lines

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any

S4_SIX_SITE_VALUE_SCHEMA = "boundflow.asplos27-s4-six-site-value/v1"
S4_SIX_SITE_VALUE_FUNCTION = "boundflow_s4_six_site_value"
S4_SIX_SITE_CUDNN_FUNCTIONS = 4
S4_SIX_SITE_CUDNN_CALLS = 6
S4_SIX_SITE_TIR_COUNT = 12
S4_SIX_SITE_READ_ARGUMENTS = 42
S4_SIX_SITE_WRITE_TARGETS = 7
S4_SIX_SITE_ARGUMENTS = 49
S4_SIX_SITE_QNAN_BITS = 0x7FC00000
S4_SELECTOR_NONFINITE_MASK = 0x7F800000
S4_SELECTOR_PACK_SPECS = (
    ("endpoint_ainput_v2", "boundflow_s4_pack_ainput_ternary", 18432, "ternary"),
    ("sign_a18", "boundflow_s4_pack_a18_binary", 12288, "binary"),
    ("sign_a20", "boundflow_s4_pack_a20_binary", 6144, "binary"),
    ("sign_a24", "boundflow_s4_pack_a24_binary", 6144, "binary"),
    ("sign_a26", "boundflow_s4_pack_a26_binary", 6144, "binary"),
    ("sign_a29", "boundflow_s4_pack_a29_binary", 6144, "binary"),
)


@dataclass(frozen=True)
class S4ValueSlotV1:
    """One stable view in the caller-owned contiguous V arena."""

    name: str
    shape: tuple[int, ...]
    offset_elements: int
    length_elements: int


S4_VALUE_SLOTS_V1 = (
    S4ValueSlotV1("V17", (6, 8, 16, 16), 0, 12288),
    S4ValueSlotV1("V19", (6, 16, 8, 8), 12288, 6144),
    S4ValueSlotV1("V23", (6, 16, 8, 8), 18432, 6144),
    S4ValueSlotV1("V25", (6, 16, 8, 8), 24576, 6144),
    S4ValueSlotV1("V28", (6, 16, 8, 8), 30720, 6144),
    S4ValueSlotV1("V31", (6, 100), 36864, 600),
)

S4_VALUE_ARENA_ELEMENTS = 37464
S4_VALUE_ARENA_BYTES = 149856


@dataclass(frozen=True)
class CompiledS4SelectorPackV1:
    """Six exact selector pack symbols and content-derived identities."""

    executable: Any
    unscheduled_tir_json: str
    scheduled_tir_json: str
    device_sources: tuple[str, ...]
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hashes: tuple[str, ...]
    exported_symbols: tuple[str, ...]
    target: str
    performance_claimed: bool = False

    def validate(self) -> None:
        expected_symbols = tuple(item[1] for item in S4_SELECTOR_PACK_SPECS)
        if (
            hashlib.sha256(self.unscheduled_tir_json.encode()).hexdigest()
            != self.unscheduled_tir_hash
            or hashlib.sha256(self.scheduled_tir_json.encode()).hexdigest()
            != self.scheduled_tir_hash
            or tuple(
                hashlib.sha256(value.encode()).hexdigest()
                for value in self.device_sources
            )
            != self.device_source_hashes
            or not self.device_sources
            or self.exported_symbols != expected_symbols
            or not self.target
            or self.performance_claimed
        ):
            raise ValueError("S4 selector pack compiled identity differs")


def _selector_pack_primfunc(symbol: str, numel: int, policy: str):
    import tvm
    from tvm import te

    source = te.placeholder((numel,), "float32", name="coefficient")
    mask = tvm.tir.const(S4_SELECTOR_NONFINITE_MASK, "uint32")
    zero = tvm.tir.const(0.0, "float32")

    def classify(index: Any) -> Any:
        bits = tvm.tir.reinterpret("uint32", source[index])
        nonfinite = tvm.tir.bitwise_and(bits, mask) == mask
        if policy == "ternary":
            finite_value = tvm.tir.if_then_else(
                source[index] > zero,
                tvm.tir.const(1, "int8"),
                tvm.tir.if_then_else(
                    source[index] < zero,
                    tvm.tir.const(-1, "int8"),
                    tvm.tir.const(0, "int8"),
                ),
            )
        else:
            finite_value = tvm.tir.if_then_else(
                source[index] >= zero,
                tvm.tir.const(1, "int8"),
                tvm.tir.const(0, "int8"),
            )
        return tvm.tir.if_then_else(
            nonfinite, tvm.tir.const(-128, "int8"), finite_value
        )

    selector = te.compute((numel,), classify, name="packed_selector")
    return (
        te.create_prim_func([source, selector])
        .with_attr("global_symbol", symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", S4_SIX_SITE_VALUE_SCHEMA)
        .with_attr("boundflow.selector_policy", policy)
    )


def build_s4_selector_pack_tir_modules_v1() -> tuple[Any, Any]:
    """Build six unscheduled and fixed-schedule selector pack functions."""

    import tvm

    unscheduled = tvm.IRModule(
        {
            symbol: _selector_pack_primfunc(symbol, numel, policy)
            for _name, symbol, numel, policy in S4_SELECTOR_PACK_SPECS
        }
    )
    scheduled = tvm.IRModule(
        {
            symbol: _schedule_elementwise(
                tvm, symbol, unscheduled[symbol], "packed_selector"
            ).with_attr("global_symbol", symbol)
            for _name, symbol, _numel, _policy in S4_SELECTOR_PACK_SPECS
        }
    )
    return unscheduled, scheduled


def compile_s4_selector_pack_v1(*, device_index: int = 0) -> CompiledS4SelectorPackV1:
    """Compile the six selector pack kernels with no fallback."""

    import torch
    import tvm

    unscheduled, scheduled = build_s4_selector_pack_tir_modules_v1()
    unscheduled_json = tvm.ir.save_json(unscheduled)
    scheduled_json = tvm.ir.save_json(scheduled)
    major, minor = torch.cuda.get_device_capability(device_index)
    target = f"cuda -arch=sm_{major}{minor}"
    executable = tvm.compile(scheduled, target=target)
    sources = tuple(module.inspect_source() for module in executable.mod.imports)
    compiled = CompiledS4SelectorPackV1(
        executable=executable,
        unscheduled_tir_json=unscheduled_json,
        scheduled_tir_json=scheduled_json,
        device_sources=sources,
        unscheduled_tir_hash=hashlib.sha256(unscheduled_json.encode()).hexdigest(),
        scheduled_tir_hash=hashlib.sha256(scheduled_json.encode()).hexdigest(),
        device_source_hashes=tuple(
            hashlib.sha256(value.encode()).hexdigest() for value in sources
        ),
        exported_symbols=tuple(item[1] for item in S4_SELECTOR_PACK_SPECS),
        target=target,
    )
    compiled.validate()
    return compiled


@dataclass(frozen=True)
class CompiledS4SixSiteValueV1:
    """Compiled identity for the exact six-site selected-value graph."""

    executable: Any
    function_name: str
    source_relax_ir_json: str
    partitioned_relax_ir_json: str
    lowered_relax_ir_json: str
    device_sources: tuple[str, ...]
    source_relax_ir_hash: str
    partitioned_relax_ir_hash: str
    lowered_relax_ir_hash: str
    device_source_hashes: tuple[str, ...]
    target: str
    cudnn_partition_function_count: int
    cudnn_conv_call_count: int
    selected_tir_count: int
    compile_ms: float
    performance_claimed: bool = False

    def validate(self) -> None:
        hashes = (
            self.source_relax_ir_hash,
            self.partitioned_relax_ir_hash,
            self.lowered_relax_ir_hash,
            *self.device_source_hashes,
        )
        if (
            self.function_name != S4_SIX_SITE_VALUE_FUNCTION
            or any(len(value) != 64 for value in hashes)
            or not self.device_source_hashes
            or hashlib.sha256(self.source_relax_ir_json.encode()).hexdigest()
            != self.source_relax_ir_hash
            or hashlib.sha256(self.partitioned_relax_ir_json.encode()).hexdigest()
            != self.partitioned_relax_ir_hash
            or hashlib.sha256(self.lowered_relax_ir_json.encode()).hexdigest()
            != self.lowered_relax_ir_hash
            or tuple(
                hashlib.sha256(value.encode()).hexdigest()
                for value in self.device_sources
            )
            != self.device_source_hashes
            or not self.target
            or self.cudnn_partition_function_count != S4_SIX_SITE_CUDNN_FUNCTIONS
            or self.cudnn_conv_call_count != S4_SIX_SITE_CUDNN_CALLS
            or self.selected_tir_count != S4_SIX_SITE_TIR_COUNT
            or self.compile_ms <= 0.0
            or self.performance_claimed
        ):
            raise ValueError("S4 six-site compiled identity differs")


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
    selector = te.placeholder((18432,), "int8", name="endpoint_selector")
    half = tvm.tir.const(0.5, "float32")
    qnan = tvm.tir.reinterpret(
        "float32", tvm.tir.const(S4_SIX_SITE_QNAN_BITS, "uint32")
    )

    def selected(d: Any, c: Any, h: Any, w: Any) -> Any:
        flat = d * 3072 + c * 1024 + h * 32 + w
        return tvm.tir.if_then_else(
            selector[flat] == tvm.tir.const(1, "int8"),
            lower[d, c, h, w],
            tvm.tir.if_then_else(
                selector[flat] == tvm.tir.const(-1, "int8"),
                upper[d, c, h, w],
                tvm.tir.if_then_else(
                    selector[flat] == tvm.tir.const(0, "int8"),
                    (lower[d, c, h, w] + upper[d, c, h, w]) * half,
                    qnan,
                ),
            ),
        )

    output = te.compute((6, 3, 32, 32), selected, name="selected_input")
    return te.create_prim_func([lower, upper, selector, output]).with_attr(
        "boundflow.schema_version", S4_SIX_SITE_VALUE_SCHEMA
    )


def _selected_relu_primfunc(
    *, channels: int, height: int, width: int, alpha_width: int
):
    import tvm
    from tvm import te

    pre = te.placeholder((6, channels, height, width), "float32", name="pre")
    selector = te.placeholder(
        (6 * channels * height * width,), "int8", name="incoming_selector"
    )
    lower = te.placeholder((6, channels, height, width), "float32", name="lower")
    upper = te.placeholder((6, channels, height, width), "float32", name="upper")
    alpha = te.placeholder((6, alpha_width), "float32", name="active_alpha")
    alpha_map = te.placeholder((channels * height * width,), "int32", name="alpha_map")
    zero = tvm.tir.const(0.0, "float32")
    one = tvm.tir.const(1.0, "float32")
    epsilon = tvm.tir.const(1.1920928955078125e-07, "float32")
    qnan = tvm.tir.reinterpret(
        "float32", tvm.tir.const(S4_SIX_SITE_QNAN_BITS, "uint32")
    )

    def selected(d: Any, c: Any, h: Any, w: Any) -> Any:
        feature = c * height * width + h * width + w
        flat = d * channels * height * width + feature
        lookup = alpha_map[feature]
        compact = alpha[d, tvm.tir.max(lookup, 0)]
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
        valid = tvm.tir.any(
            selector[flat] == tvm.tir.const(0, "int8"),
            selector[flat] == tvm.tir.const(1, "int8"),
        )
        slope = tvm.tir.if_then_else(selector[flat] == 1, lower_slope, upper_slope)
        intercept = tvm.tir.if_then_else(
            tvm.tir.all(selector[flat] == 0, ambiguous),
            -lower[d, c, h, w] * upper_slope,
            zero,
        )
        value = pre[d, c, h, w] * slope + intercept
        return tvm.tir.if_then_else(valid, value, qnan)

    output = te.compute((6, channels, height, width), selected, name="selected_relu")
    return te.create_prim_func(
        [pre, selector, lower, upper, alpha, alpha_map, output]
    ).with_attr("boundflow.schema_version", S4_SIX_SITE_VALUE_SCHEMA)


def _copy_primfunc(shape: tuple[int, ...]):
    from tvm import te

    source = te.placeholder(shape, "float32", name="source")
    copied = te.compute(shape, lambda *indices: source[indices], name="persistent_copy")
    return te.create_prim_func([source, copied]).with_attr(
        "boundflow.schema_version", S4_SIX_SITE_VALUE_SCHEMA
    )


def _tensor_var(relax: Any, name: str, shape: tuple[int, ...], dtype: str = "float32"):
    return relax.Var(name, relax.TensorStructInfo(shape, dtype))


def build_s4_six_site_value_relax_module_v1():
    """Build the exact six-site graph with caller-owned in-place outputs."""

    import tvm
    from tvm import relax

    builder = relax.BlockBuilder()
    parameter_specs = (
        ("input_lower", (6, 3, 32, 32), "float32"),
        ("input_upper", (6, 3, 32, 32), "float32"),
        ("endpoint_selector", (18432,), "int8"),
        ("weight0", (8, 3, 3, 3), "float32"),
        ("bias0", (8,), "float32"),
        ("lower17", (6, 8, 16, 16), "float32"),
        ("upper17", (6, 8, 16, 16), "float32"),
        ("alpha17", (6, 164), "float32"),
        ("alpha_map17", (2048,), "int32"),
        ("sign_a18", (12288,), "int8"),
        ("weight2", (16, 8, 3, 3), "float32"),
        ("bias2", (16,), "float32"),
        ("lower19", (6, 16, 8, 8), "float32"),
        ("upper19", (6, 16, 8, 8), "float32"),
        ("alpha19", (6, 132), "float32"),
        ("alpha_map19", (1024,), "int32"),
        ("sign_a20", (6144,), "int8"),
        ("weight4", (16, 16, 3, 3), "float32"),
        ("bias4", (16,), "float32"),
        ("weight5", (16, 8, 1, 1), "float32"),
        ("bias5", (16,), "float32"),
        ("lower23", (6, 16, 8, 8), "float32"),
        ("upper23", (6, 16, 8, 8), "float32"),
        ("alpha23", (6, 121), "float32"),
        ("alpha_map23", (1024,), "int32"),
        ("sign_a24", (6144,), "int8"),
        ("weight8", (16, 16, 3, 3), "float32"),
        ("bias8", (16,), "float32"),
        ("lower25", (6, 16, 8, 8), "float32"),
        ("upper25", (6, 16, 8, 8), "float32"),
        ("alpha25", (6, 86), "float32"),
        ("alpha_map25", (1024,), "int32"),
        ("sign_a26", (6144,), "int8"),
        ("weight10", (16, 16, 3, 3), "float32"),
        ("bias10", (16,), "float32"),
        ("lower28", (6, 16, 8, 8), "float32"),
        ("upper28", (6, 16, 8, 8), "float32"),
        ("alpha28", (6, 178), "float32"),
        ("alpha_map28", (1024,), "int32"),
        ("sign_a29", (6144,), "int8"),
        ("weight14", (100, 1024), "float32"),
        ("bias14", (100,), "float32"),
        ("selected_input_target", (6, 3, 32, 32), "float32"),
        ("v17_target", (6, 8, 16, 16), "float32"),
        ("v19_target", (6, 16, 8, 8), "float32"),
        ("v23_target", (6, 16, 8, 8), "float32"),
        ("v25_target", (6, 16, 8, 8), "float32"),
        ("v28_target", (6, 16, 8, 8), "float32"),
        ("v31_target", (6, 100), "float32"),
    )
    parameters = [
        _tensor_var(relax, name, shape, dtype) for name, shape, dtype in parameter_specs
    ]
    if len(parameters) != S4_SIX_SITE_ARGUMENTS:
        raise AssertionError("S4 six-site ABI parameter count differs")

    def add_tir(symbol: str, primfunc: Any, block_name: str):
        return builder.add_func(
            _schedule_elementwise(tvm, symbol, primfunc, block_name).with_attr(
                "global_symbol", symbol
            ),
            symbol,
        )

    input_global = add_tir(
        "boundflow_s4_select_input_ternary_tir",
        _input_select_primfunc(),
        "selected_input",
    )
    relu_specs = (
        ("relu17", 8, 16, 16, 164),
        ("relu19", 16, 8, 8, 132),
        ("relu23", 16, 8, 8, 121),
        ("relu25", 16, 8, 8, 86),
        ("relu28", 16, 8, 8, 178),
    )
    relu_globals = {
        name: add_tir(
            f"boundflow_s4_select_{name}_tir",
            _selected_relu_primfunc(
                channels=channels,
                height=height,
                width=width,
                alpha_width=alpha_width,
            ),
            "selected_relu",
        )
        for name, channels, height, width, alpha_width in relu_specs
    }
    copy_shapes = (
        ("v17", (6, 8, 16, 16)),
        ("v19", (6, 16, 8, 8)),
        ("v23", (6, 16, 8, 8)),
        ("v25", (6, 16, 8, 8)),
        ("v28", (6, 16, 8, 8)),
        ("v31", (6, 100)),
    )
    copy_globals = {
        name: add_tir(
            f"boundflow_s4_copy_{name}_tir", _copy_primfunc(shape), "persistent_copy"
        )
        for name, shape in copy_shapes
    }

    def conv_bias(
        data: Any,
        weight: Any,
        bias: Any,
        *,
        strides: tuple[int, int],
        padding: tuple[int, int],
        channels: int,
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

    def copy_inplace(source: Any, target: Any, global_var: Any, shape: tuple[int, ...]):
        return builder.emit(
            relax.call_tir_inplace(
                global_var,
                relax.Tuple([source, target]),
                inplace_indices=[1],
                out_sinfo=relax.TensorStructInfo(shape, "float32"),
            )
        )

    def select_relu(
        pre: Any,
        selector: Any,
        lower: Any,
        upper: Any,
        alpha: Any,
        alpha_map: Any,
        global_var: Any,
        shape: tuple[int, ...],
    ):
        return builder.emit(
            relax.call_tir(
                global_var,
                relax.Tuple([pre, selector, lower, upper, alpha, alpha_map]),
                out_sinfo=relax.TensorStructInfo(shape, "float32"),
            )
        )

    with builder.function(S4_SIX_SITE_VALUE_FUNCTION, parameters):
        with builder.dataflow():
            selected_input = builder.emit(
                relax.call_tir_inplace(
                    input_global,
                    relax.Tuple(
                        [parameters[0], parameters[1], parameters[2], parameters[42]]
                    ),
                    inplace_indices=[3],
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
            v17 = copy_inplace(
                pre17, parameters[43], copy_globals["v17"], (6, 8, 16, 16)
            )
            selected17 = select_relu(
                v17,
                parameters[9],
                parameters[5],
                parameters[6],
                parameters[7],
                parameters[8],
                relu_globals["relu17"],
                (6, 8, 16, 16),
            )

            pre19 = conv_bias(
                selected17,
                parameters[10],
                parameters[11],
                strides=(2, 2),
                padding=(1, 1),
                channels=16,
            )
            v19 = copy_inplace(
                pre19, parameters[44], copy_globals["v19"], (6, 16, 8, 8)
            )
            selected19 = select_relu(
                v19,
                parameters[16],
                parameters[12],
                parameters[13],
                parameters[14],
                parameters[15],
                relu_globals["relu19"],
                (6, 16, 8, 8),
            )

            main23 = conv_bias(
                selected19,
                parameters[17],
                parameters[18],
                strides=(1, 1),
                padding=(1, 1),
                channels=16,
            )
            shortcut23 = conv_bias(
                selected17,
                parameters[19],
                parameters[20],
                strides=(2, 2),
                padding=(0, 0),
                channels=16,
            )
            pre23 = builder.emit(relax.op.add(main23, shortcut23))
            v23 = copy_inplace(
                pre23, parameters[45], copy_globals["v23"], (6, 16, 8, 8)
            )
            selected23 = select_relu(
                v23,
                parameters[25],
                parameters[21],
                parameters[22],
                parameters[23],
                parameters[24],
                relu_globals["relu23"],
                (6, 16, 8, 8),
            )

            pre25 = conv_bias(
                selected23,
                parameters[26],
                parameters[27],
                strides=(1, 1),
                padding=(1, 1),
                channels=16,
            )
            v25 = copy_inplace(
                pre25, parameters[46], copy_globals["v25"], (6, 16, 8, 8)
            )
            selected25 = select_relu(
                v25,
                parameters[32],
                parameters[28],
                parameters[29],
                parameters[30],
                parameters[31],
                relu_globals["relu25"],
                (6, 16, 8, 8),
            )

            main28 = conv_bias(
                selected25,
                parameters[33],
                parameters[34],
                strides=(1, 1),
                padding=(1, 1),
                channels=16,
            )
            pre28 = builder.emit(relax.op.add(main28, selected23))
            v28 = copy_inplace(
                pre28, parameters[47], copy_globals["v28"], (6, 16, 8, 8)
            )
            selected28 = select_relu(
                v28,
                parameters[39],
                parameters[35],
                parameters[36],
                parameters[37],
                parameters[38],
                relu_globals["relu28"],
                (6, 16, 8, 8),
            )

            flattened = builder.emit(relax.op.reshape(selected28, (6, 1024)))
            transposed_weight = builder.emit(
                relax.op.permute_dims(parameters[40], axes=(1, 0))
            )
            linear = builder.emit(
                relax.op.matmul(flattened, transposed_weight, out_dtype="float32")
            )
            pre31 = builder.emit(relax.op.add(linear, parameters[41]))
            v31 = copy_inplace(pre31, parameters[48], copy_globals["v31"], (6, 100))
            output = builder.emit_output(relax.Tuple([v17, v19, v23, v25, v28, v31]))
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


def compile_s4_six_site_value_v1(*, device_index: int = 0) -> CompiledS4SixSiteValueV1:
    """Compile the exact S4-1B graph without runtime fallback."""

    import torch
    import tvm
    from tvm import dlight as dl
    from tvm import relax, transform
    from tvm.relax.backend.cuda.cudnn import partition_for_cudnn

    source = build_s4_six_site_value_relax_module_v1()
    source_json = tvm.ir.save_json(source)
    source_hash = hashlib.sha256(source_json.encode()).hexdigest()
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
        partitioned[S4_SIX_SITE_VALUE_FUNCTION].body, count_cudnn_calls
    )
    if (
        partition_function_count != S4_SIX_SITE_CUDNN_FUNCTIONS
        or cudnn_call_count != S4_SIX_SITE_CUDNN_CALLS
    ):
        raise RuntimeError(
            "S4 six-site cuDNN partition/call count differs: "
            f"{partition_function_count}/{cudnn_call_count}"
        )
    partitioned_json = tvm.ir.save_json(partitioned)
    partitioned_hash = hashlib.sha256(partitioned_json.encode()).hexdigest()
    lowered = relax.transform.RunCodegen()(partitioned)
    lowered_json = tvm.ir.save_json(lowered)
    lowered_hash = hashlib.sha256(lowered_json.encode()).hexdigest()
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
        sources = (lowered_json,)
    compiled = CompiledS4SixSiteValueV1(
        executable=executable,
        function_name=S4_SIX_SITE_VALUE_FUNCTION,
        source_relax_ir_json=source_json,
        partitioned_relax_ir_json=partitioned_json,
        lowered_relax_ir_json=lowered_json,
        device_sources=sources,
        source_relax_ir_hash=source_hash,
        partitioned_relax_ir_hash=partitioned_hash,
        lowered_relax_ir_hash=lowered_hash,
        device_source_hashes=tuple(
            hashlib.sha256(value.encode()).hexdigest() for value in sources
        ),
        target=str(target),
        cudnn_partition_function_count=partition_function_count,
        cudnn_conv_call_count=cudnn_call_count,
        selected_tir_count=S4_SIX_SITE_TIR_COUNT,
        compile_ms=compile_ms,
    )
    compiled.validate()
    return compiled


__all__ = [
    "CompiledS4SelectorPackV1",
    "CompiledS4SixSiteValueV1",
    "S4_SIX_SITE_ARGUMENTS",
    "S4_SIX_SITE_CUDNN_CALLS",
    "S4_SIX_SITE_CUDNN_FUNCTIONS",
    "S4_SIX_SITE_VALUE_FUNCTION",
    "S4_SIX_SITE_QNAN_BITS",
    "S4_SIX_SITE_READ_ARGUMENTS",
    "S4_SIX_SITE_TIR_COUNT",
    "S4_SIX_SITE_VALUE_SCHEMA",
    "S4_SIX_SITE_WRITE_TARGETS",
    "S4_SELECTOR_PACK_SPECS",
    "S4_VALUE_ARENA_BYTES",
    "S4_VALUE_ARENA_ELEMENTS",
    "S4_VALUE_SLOTS_V1",
    "S4ValueSlotV1",
    "build_s4_six_site_value_relax_module_v1",
    "build_s4_selector_pack_tir_modules_v1",
    "compile_s4_six_site_value_v1",
    "compile_s4_selector_pack_v1",
]
