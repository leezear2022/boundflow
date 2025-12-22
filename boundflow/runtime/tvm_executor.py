from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import torch

from ..domains.interval import IntervalDomain, IntervalState
from ..ir.task import BFTaskModule, TaskKind
from ..ir.task import StoragePlan
from ..ir.task import TaskOp
from ..runtime.task_executor import LinfInputSpec
from ..backends.tvm.interval_linear import IntervalLinearKey, build_interval_linear_module
from ..backends.tvm.interval_conv2d import IntervalConv2dKey, build_interval_conv2d_module
from ..backends.tvm.relax_interval_linear import build_relax_interval_linear_vm_exec
from ..backends.tvm.relax_interval_conv2d import build_relax_interval_conv2d_vm_exec
from ..backends.tvm.relax_task_lowering import RelaxLoweringMode, build_interval_linear_relax_ir_module


class MemoryPlanMode(Enum):
    DEFAULT = "default"
    DISABLE_STATIC_PLAN = "disable_static_plan"
    FORCE_STATIC_PLAN = "force_static_plan"


@dataclass(frozen=True)
class TVMExecutorOptions:
    target: Optional[str] = None  # e.g. "llvm" or "cuda"
    kernel_style: str = "relax"  # "relax" (preferred), "call_tir", or "te" (legacy demo)
    # PR#10: compile-side observability.
    enable_pass_timing: bool = False
    enable_dump_ir: bool = False
    dump_ir_dir: str = ".benchmarks/tvm_ir"
    dump_ir_refresh: bool = False
    # Optional tag to avoid accidental cache collisions across configs.
    compile_cache_tag: str = ""
    # Optional on-disk compile cache for Relax VM executables (cross-process reuse).
    # When enabled, task-level RELAX_OPS compilation can load/store a shared library + json spec.
    compile_cache_dir: str = ""
    compile_cache_refresh: bool = False
    # PR#11A: run whole task as a single Relax function (RELAX_OPS lowering) when possible.
    enable_task_relax_ops: bool = False
    # PR#11B: fusion pipeline control (RELAX_OPS path).
    enable_task_fusion_pipeline: bool = False
    task_fuse_opt_level: int = -1
    # PR#11C: reduce VM call overhead and provide optional VM-level optimization passes.
    enable_vm_cache: bool = True
    enable_vm_packed_func_cache: bool = True
    task_vm_opt_passes: tuple[str, ...] = ()
    # PR#12: control Relax static memory planning.
    memory_plan_mode: MemoryPlanMode = MemoryPlanMode.DEFAULT
    # PR#12 follow-up: reserve dynamic shape upper bounds for StaticPlanBlockMemory (not wired yet).
    tir_var_upper_bound: Optional[Dict[str, int]] = None


@dataclass
class TVMRunStats:
    tvm_ops: List[str]
    fallback_ops: List[str]
    linear_kernel_cache: Dict[str, int]
    conv2d_kernel_cache: Dict[str, int]
    kernel_style: str


class TVMTaskExecutor:
    """
    v0: Python driver + TVM compiled kernels.
    目标是打通“同一个 BFTaskModule：PythonTaskExecutor vs TVMTaskExecutor 输出一致”的通路，
    暂不把复杂编排/控制流塞进 Relax VM。
    """

    def __init__(self, *, options: Optional[TVMExecutorOptions] = None):
        self.options = options or TVMExecutorOptions()
        self.domain = IntervalDomain()
        self.last_stats: Optional[TVMRunStats] = None
        # (kernel_style, IntervalLinearKey) -> relax.Executable
        self._linear_exec_cache: Dict[tuple[str, IntervalLinearKey], Any] = {}
        # task_cache_key_hash -> {"ex": Executable, "spec": IntervalTaskLoweringSpec}
        self._task_exec_cache: Dict[str, Dict[str, Any]] = {}
        # (cache_key_hash, dev_type_name, dev_index) -> {"vm": VirtualMachine, "fn": PackedFunc}
        self._vm_cache: Dict[tuple[str, str, int], Dict[str, Any]] = {}
        # cache_key -> compile stats (jsonable)
        self._compile_stats: Dict[str, Dict[str, Any]] = {}
        # PR#13B: compile cache accounting for fairness in bench.
        self._task_compile_cache_hit: int = 0
        self._task_compile_cache_miss: int = 0
        self._task_compile_fail: int = 0

    def _select_target(self) -> str:
        import tvm

        if self.options.target:
            return self.options.target
        if tvm.runtime.enabled("llvm"):
            return "llvm"
        if tvm.runtime.enabled("cuda"):
            return "cuda"
        return "llvm"

    @staticmethod
    def _parse_pass_timing_render(text: str) -> List[Dict[str, Any]]:
        """
        Best-effort parser for tvm.ir.instrument.PassTimingInstrument.render() output.
        """
        rows: List[Dict[str, Any]] = []
        for line in (text or "").splitlines():
            s = line.strip()
            if not s or s.startswith("Name") or s.startswith("---"):
                continue
            # Current TVM render format (C++): "<pass_name>: <dur>us [self] (..)"
            if ":" not in s:
                continue
            name, rest = s.split(":", 1)
            name = name.strip()
            rest = rest.strip()
            if not rest.endswith(")") and "us" not in rest:
                continue
            # First token is like "123us"
            tok = rest.split()[0]
            if not tok.endswith("us"):
                continue
            try:
                us = float(tok[:-2])
            except Exception:
                continue
            rows.append({"pass": name, "time_ms": us / 1e3})
        return rows

    def get_compile_stats(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._compile_stats)

    def get_task_compile_cache_stats(self) -> Dict[str, int]:
        return {
            "task_compile_cache_hit": int(self._task_compile_cache_hit),
            "task_compile_cache_miss": int(self._task_compile_cache_miss),
            "task_compile_fail": int(self._task_compile_fail),
        }

    def _get_vm_callable(self, *, cache_key_hash: str, ex: Any, dev: Any, func_name: str) -> tuple[Any, Any]:
        """
        Get (vm, callable) for a compiled executable on a specific device.

        callable is either a cached PackedFunc (preferred) or vm[func_name] retrieved on demand.
        """
        from tvm import relax  # noqa: PLC0415

        dev_key = (str(cache_key_hash), str(getattr(dev, "type", "unknown")), int(getattr(dev, "index", 0)))
        if self.options.enable_vm_cache and dev_key in self._vm_cache:
            entry = self._vm_cache[dev_key]
            vm = entry.get("vm")
            fn = entry.get("fn")
            if vm is not None:
                if fn is None and self.options.enable_vm_packed_func_cache:
                    fn = vm[func_name]
                    entry["fn"] = fn
                return vm, (fn or vm[func_name])

        vm = relax.VirtualMachine(ex, dev)
        fn = vm[func_name] if self.options.enable_vm_packed_func_cache else None
        if self.options.enable_vm_cache:
            self._vm_cache[dev_key] = {"vm": vm, "fn": fn}
        return vm, (fn or vm[func_name])

    def _compile_interval_linear_executable(self, key: IntervalLinearKey):
        import tvm
        from tvm import relax

        k = (str(self.options.kernel_style), str(self.options.compile_cache_tag), key)
        if k in self._linear_exec_cache:
            return self._linear_exec_cache[k]

        if self.options.kernel_style == "call_tir":
            ir_mod = build_interval_linear_relax_ir_module(key, mode=RelaxLoweringMode.CALL_TIR, relax_func_name="main")
            compile_mode = "call_tir"
        elif self.options.kernel_style == "relax":
            ir_mod = build_interval_linear_relax_ir_module(key, mode=RelaxLoweringMode.RELAX_OPS, relax_func_name="main")
            compile_mode = "relax_ops"
        else:
            raise ValueError(f"unsupported kernel_style for relax executable: {self.options.kernel_style}")

        cache_key_str = repr(k).encode("utf-8")
        cache_key_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]

        instruments: List[Any] = []
        timing_inst = None
        dump_dir = None
        if self.options.enable_pass_timing:
            from tvm.ir.instrument import PassTimingInstrument  # noqa: PLC0415

            timing_inst = PassTimingInstrument()
            instruments.append(timing_inst)
        if self.options.enable_dump_ir:
            from tvm.ir.instrument import DumpIR  # noqa: PLC0415

            run_id = os.environ.get("BOUNDFLOW_TVM_RUN_ID") or f"{int(time.time())}_{os.getpid()}"
            dump_dir = os.path.join(str(self.options.dump_ir_dir), str(run_id), cache_key_hash)
            instruments.append(DumpIR(dump_dir=dump_dir, refresh=bool(self.options.dump_ir_refresh)))

        t0 = time.perf_counter_ns()
        if instruments:
            from tvm import transform  # noqa: PLC0415

            with transform.PassContext(instruments=instruments):
                ex = relax.build(ir_mod, target=key.target)
                # PassTimingInstrument clears profiles on exit_pass_ctx, so render before leaving the context.
                rendered = str(timing_inst.render()) if timing_inst is not None else None
        else:
            ex = relax.build(ir_mod, target=key.target)
            rendered = None
        t1 = time.perf_counter_ns()

        stats: Dict[str, Any] = {
            "cache_key_hash": cache_key_hash,
            "kernel_style": str(self.options.kernel_style),
            "compile_mode": compile_mode,
            "target": str(key.target),
            "key": {
                "batch": int(key.batch),
                "in_features": int(key.in_features),
                "out_features": int(key.out_features),
                "dtype": str(key.dtype),
            },
            "compile_ms": (t1 - t0) / 1e6,
            "pass_timing": None,
            "pass_timing_render": None,
            "dump_ir_dir": dump_dir,
        }
        if timing_inst is not None:
            stats["pass_timing_render"] = rendered
            stats["pass_timing"] = self._parse_pass_timing_render(rendered or "")
        self._compile_stats[cache_key_hash] = stats

        self._linear_exec_cache[k] = ex
        return ex

    def _compile_interval_task_relax_ops(self, task, *, storage_plan: StoragePlan, target: str) -> tuple[Any, Any, str]:
        """
        Compile a whole task into a single Relax function using RELAX_OPS lowering.

        Returns (executable, lowering_spec, cache_key_hash)
        """
        from ..backends.tvm.relax_interval_task_ops import build_interval_task_relax_ops_ir_module  # noqa: PLC0415

        # Build a stable signature for caching.
        sig = {
            "task_id": str(getattr(task, "task_id", "")),
            "op_types": [str(getattr(op, "op_type", "")) for op in getattr(task, "ops", [])],
            "inputs": list(getattr(task, "input_values", [])),
            "outputs": list(getattr(task, "output_values", [])),
            "params": sorted(list(getattr(task, "params", []) or [])),
            "target": str(target),
            "kernel_style": str(self.options.kernel_style),
            "compile_cache_tag": str(self.options.compile_cache_tag),
            "enable_task_fusion_pipeline": bool(self.options.enable_task_fusion_pipeline),
            "task_fuse_opt_level": int(self.options.task_fuse_opt_level),
            "task_vm_opt_passes": tuple(self.options.task_vm_opt_passes or ()),
            "memory_plan_mode": str(getattr(self.options.memory_plan_mode, "value", self.options.memory_plan_mode)),
            "tir_var_upper_bound": dict(self.options.tir_var_upper_bound or {}),
        }
        cache_key_hash = hashlib.sha256(repr(sig).encode("utf-8")).hexdigest()[:16]
        if cache_key_hash in self._task_exec_cache:
            self._task_compile_cache_hit += 1
            cached = self._task_exec_cache[cache_key_hash]
            return cached["ex"], cached["spec"], cache_key_hash

        # Optional cross-process disk cache: try load a previously compiled executable + lowering spec.
        cache_dir = str(self.options.compile_cache_dir or "")
        if cache_dir and (not bool(self.options.compile_cache_refresh)):
            try:
                os.makedirs(cache_dir, exist_ok=True)
                lib_path = os.path.join(cache_dir, f"task_{cache_key_hash}.so")
                spec_path = os.path.join(cache_dir, f"task_{cache_key_hash}.spec.json")
                if os.path.exists(lib_path) and os.path.exists(spec_path):
                    import tvm

                    from ..backends.tvm.relax_interval_task_ops import IntervalTaskLoweringSpec  # noqa: PLC0415

                    with open(spec_path, "r", encoding="utf-8") as f:
                        spec_dict = json.load(f)
                    spec = IntervalTaskLoweringSpec(**spec_dict)
                    ex = tvm.runtime.load_module(lib_path)
                    stats: Dict[str, Any] = {
                        "cache_key_hash": cache_key_hash,
                        "kind": "task_relax_ops",
                        "target": str(target),
                        "compile_ms": 0.0,
                        "pass_timing": None,
                        "pass_timing_render": None,
                        "dump_ir_dir": None,
                        "memory_plan_mode": str(getattr(self.options.memory_plan_mode, "value", self.options.memory_plan_mode)),
                        "memory_stats": None,
                        "tir_var_upper_bound": dict(self.options.tir_var_upper_bound or {}),
                        "task_id": str(getattr(task, "task_id", "")),
                        "op_types": [str(getattr(op, "op_type", "")) for op in getattr(task, "ops", [])],
                        "ir_stats": None,
                        "pipeline": [],
                        "task_vm_opt_passes": list(self.options.task_vm_opt_passes or ()),
                        "compile_cache_event": "disk_hit",
                        "compile_cache_dir": cache_dir,
                    }
                    self._compile_stats[cache_key_hash] = stats
                    self._task_exec_cache[cache_key_hash] = {"ex": ex, "spec": spec}
                    self._task_compile_cache_hit += 1
                    return ex, spec, cache_key_hash
            except Exception:
                # Best-effort: if disk cache is corrupted/incompatible, fall back to compiling.
                pass

        ir_mod, spec = build_interval_task_relax_ops_ir_module(
            task,
            storage_plan=storage_plan,
            target=target,
            func_name="main",
        )
        # If provided, annotate Relax function with dynamic shape upper bounds for StaticPlanBlockMemory.
        if self.options.tir_var_upper_bound:
            try:
                import tvm  # noqa: PLC0415

                mod2 = tvm.IRModule(ir_mod.functions)
                main_gv = mod2.get_global_var("main")
                func = mod2[main_gv]
                mod2.update_func(main_gv, func.with_attr("tir_var_upper_bound", dict(self.options.tir_var_upper_bound)))
                ir_mod = mod2
            except Exception:
                # Best-effort: do not block compilation if attr update fails.
                pass

        import tvm
        from tvm import relax
        from tvm import transform

        from ..backends.tvm.relax_analysis import collect_relax_ir_stats  # noqa: PLC0415

        instruments: List[Any] = []
        timing_inst = None
        dump_dir = None
        if self.options.enable_pass_timing:
            from tvm.ir.instrument import PassTimingInstrument  # noqa: PLC0415

            timing_inst = PassTimingInstrument()
            instruments.append(timing_inst)
        if self.options.enable_dump_ir:
            from tvm.ir.instrument import DumpIR  # noqa: PLC0415

            run_id = os.environ.get("BOUNDFLOW_TVM_RUN_ID") or f"{int(time.time())}_{os.getpid()}"
            dump_dir = os.path.join(str(self.options.dump_ir_dir), str(run_id), cache_key_hash)
            instruments.append(DumpIR(dump_dir=dump_dir, refresh=bool(self.options.dump_ir_refresh)))

        # Build a deterministic and controllable Relax pipeline for task-level compilation.
        #
        # Important: the Relax VM codegen requires the default build pipeline passes
        # (e.g. LowerAllocTensor/LowerRuntimeBuiltin/AttachGlobalSymbol). If we bypass them,
        # we may hit VM codegen errors like "cannot handle relax.builtin.alloc_tensor".
        #
        # Therefore we compose:
        # - (optional) a fusion pre-pass stage to reduce call_tir count
        # - the official default build pipeline (lowering+memory planning+runtime lowering)
        pre_passes: List[Any] = []
        if self.options.enable_task_fusion_pipeline:
            pre_passes += [
                relax.transform.Normalize(),
                relax.transform.FoldConstant(),
                relax.transform.LegalizeOps(),
                relax.transform.ConvertToDataflow(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FuseOps(fuse_opt_level=int(self.options.task_fuse_opt_level)),
                relax.transform.FuseTIR(),
                relax.transform.DeadCodeElimination(),
                relax.transform.RemoveUnusedOutputs(),
            ]

        # Optional VM-level optimization passes (kept as strings for JSON-ability).
        for p in (self.options.task_vm_opt_passes or ()):
            if not hasattr(relax.transform, str(p)):
                raise ValueError(f"unknown relax pass in task_vm_opt_passes: {p}")
            pre_passes.append(getattr(relax.transform, str(p))())

        # Compose with the official default build pipeline (required for VM codegen).
        # For DEFAULT/FORCE_STATIC_PLAN, use TVM's official default pipeline for clearer behavior boundaries.
        from tvm.relax import pipeline as relax_pipeline_mod  # noqa: PLC0415

        if self.options.memory_plan_mode in (MemoryPlanMode.DEFAULT, MemoryPlanMode.FORCE_STATIC_PLAN):
            build_pipeline_pass = relax_pipeline_mod.default_build_pipeline()
        else:
            # DISABLE_STATIC_PLAN: equivalent to default_build_pipeline but with StaticPlanBlockMemory removed.
            from tvm.relax import backend as relax_backend  # noqa: PLC0415

            default_build_passes: List[Any] = [
                relax_backend.DispatchSampling(),
                relax_backend.DispatchSortScan(),
                relax.transform.LegalizeOps(),
                relax.transform.RewriteDataflowReshape(),
                relax.transform.ToNonDataflow(),
                relax.transform.RemovePurityChecking(),
                relax.transform.CallTIRRewrite(),
                # StaticPlanBlockMemory skipped on purpose.
                relax.transform.RewriteCUDAGraph(),
                relax.transform.LowerAllocTensor(),
                relax.transform.KillAfterLastUse(),
                relax.transform.LowerRuntimeBuiltin(),
                relax.transform.ComputePrimValue(),
                relax.transform.VMShapeLower(),
                relax.transform.AttachGlobalSymbol(),
            ]
            build_pipeline_pass = transform.Sequential(default_build_passes)

        relax_pipeline = transform.Sequential(
            [
                transform.Sequential(pre_passes) if pre_passes else transform.Sequential([]),
                build_pipeline_pass,
            ]
        )

        # Collect per-stage IR stats for ablation.
        stats_ir: Dict[str, Any] = {"before": collect_relax_ir_stats(ir_mod)}
        try:
            mod_after_legalize = transform.Sequential(
                [relax.transform.Normalize(), relax.transform.FoldConstant(), relax.transform.LegalizeOps()]
            )(ir_mod)
            stats_ir["after_legalize"] = collect_relax_ir_stats(mod_after_legalize)
            if self.options.enable_task_fusion_pipeline:
                mod_after_fuse_ops = transform.Sequential(
                    [
                        relax.transform.Normalize(),
                        relax.transform.FoldConstant(),
                        relax.transform.LegalizeOps(),
                        relax.transform.ConvertToDataflow(),
                        relax.transform.AnnotateTIROpPattern(),
                        relax.transform.FuseOps(fuse_opt_level=int(self.options.task_fuse_opt_level)),
                    ]
                )(ir_mod)
                stats_ir["after_fuse_ops"] = collect_relax_ir_stats(mod_after_fuse_ops)
                mod_after_fuse_tir = transform.Sequential(
                    [
                        relax.transform.Normalize(),
                        relax.transform.FoldConstant(),
                        relax.transform.LegalizeOps(),
                        relax.transform.ConvertToDataflow(),
                        relax.transform.AnnotateTIROpPattern(),
                        relax.transform.FuseOps(fuse_opt_level=int(self.options.task_fuse_opt_level)),
                        relax.transform.FuseTIR(),
                    ]
                )(ir_mod)
                stats_ir["after_fuse_tir"] = collect_relax_ir_stats(mod_after_fuse_tir)
        except Exception:
            # Stats are best-effort; compilation should still proceed.
            pass

        from ..backends.tvm.relax_analysis import collect_relax_memory_stats  # noqa: PLC0415

        t0 = time.perf_counter_ns()
        self._task_compile_cache_miss += 1
        try:
            if instruments:
                with transform.PassContext(instruments=instruments):
                    lowered = relax_pipeline(ir_mod)
                    # (1) Structured scan stats on the lowered module (post memory planning & LowerAllocTensor).
                    by_scan = collect_relax_memory_stats(lowered)
                    # (2) TVM official estimator render (best-effort), usually run on pre-LowerAllocTensor module.
                    estimator_stage = "pre_static_plan"
                    estimate_render = None
                    try:
                        from tvm.relax import analysis as relax_analysis  # noqa: PLC0415
                        from tvm.relax import backend as relax_backend  # noqa: PLC0415

                        est_passes: List[Any] = [
                            relax_backend.DispatchSampling(),
                            relax_backend.DispatchSortScan(),
                            relax.transform.LegalizeOps(),
                            relax.transform.RewriteDataflowReshape(),
                            relax.transform.ToNonDataflow(),
                            relax.transform.RemovePurityChecking(),
                            relax.transform.CallTIRRewrite(),
                        ]
                        est_mod = transform.Sequential(est_passes)(ir_mod)
                        estimate_render = str(relax_analysis.estimate_memory_usage(est_mod))
                    except Exception:
                        pass
                    memory_stats = {
                        "by_scan": by_scan,
                        "by_tvm_estimator": estimate_render,
                        "by_tvm_estimator_stage": estimator_stage,
                    }
                    ex = relax.build(lowered, target=target, relax_pipeline=None)
                    rendered = str(timing_inst.render()) if timing_inst is not None else None
            else:
                lowered = relax_pipeline(ir_mod)
                by_scan = collect_relax_memory_stats(lowered)
                estimator_stage = "pre_static_plan"
                estimate_render = None
                try:
                    from tvm.relax import analysis as relax_analysis  # noqa: PLC0415
                    from tvm.relax import backend as relax_backend  # noqa: PLC0415

                    est_passes = [
                        relax_backend.DispatchSampling(),
                        relax_backend.DispatchSortScan(),
                        relax.transform.LegalizeOps(),
                        relax.transform.RewriteDataflowReshape(),
                        relax.transform.ToNonDataflow(),
                        relax.transform.RemovePurityChecking(),
                        relax.transform.CallTIRRewrite(),
                    ]
                    est_mod = transform.Sequential(est_passes)(ir_mod)
                    estimate_render = str(relax_analysis.estimate_memory_usage(est_mod))
                except Exception:
                    pass
                memory_stats = {
                    "by_scan": by_scan,
                    "by_tvm_estimator": estimate_render,
                    "by_tvm_estimator_stage": estimator_stage,
                }
                ex = relax.build(lowered, target=target, relax_pipeline=None)
                rendered = None
        except Exception:
            self._task_compile_fail += 1
            raise
        t1 = time.perf_counter_ns()

        stats: Dict[str, Any] = {
            "cache_key_hash": cache_key_hash,
            "kind": "task_relax_ops",
            "target": str(target),
            "compile_ms": (t1 - t0) / 1e6,
            "pass_timing": None,
            "pass_timing_render": None,
            "dump_ir_dir": dump_dir,
            "memory_plan_mode": str(getattr(self.options.memory_plan_mode, "value", self.options.memory_plan_mode)),
            "memory_stats": memory_stats,
            "tir_var_upper_bound": dict(self.options.tir_var_upper_bound or {}),
            "task_id": str(getattr(task, "task_id", "")),
            "op_types": [str(getattr(op, "op_type", "")) for op in getattr(task, "ops", [])],
            "ir_stats": stats_ir,
            "pipeline": [str(p) for p in pre_passes],
            "task_vm_opt_passes": list(self.options.task_vm_opt_passes or ()),
            "compile_cache_event": "miss",
        }
        if timing_inst is not None:
            stats["pass_timing_render"] = rendered
            stats["pass_timing"] = self._parse_pass_timing_render(rendered or "")
        self._compile_stats[cache_key_hash] = stats

        self._task_exec_cache[cache_key_hash] = {"ex": ex, "spec": spec}
        # Best-effort: write disk cache artifact for cross-process reuse.
        if cache_dir:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                lib_path = os.path.join(cache_dir, f"task_{cache_key_hash}.so")
                spec_path = os.path.join(cache_dir, f"task_{cache_key_hash}.spec.json")
                if bool(self.options.compile_cache_refresh):
                    for p in (lib_path, spec_path):
                        try:
                            if os.path.exists(p):
                                os.remove(p)
                        except Exception:
                            pass
                ex.export_library(lib_path)
                with open(spec_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "func_name": str(getattr(spec, "func_name", "main")),
                            "input_values": list(getattr(spec, "input_values", [])),
                            "param_values": list(getattr(spec, "param_values", [])),
                            "output_values": list(getattr(spec, "output_values", [])),
                            "output_flattened": bool(getattr(spec, "output_flattened", True)),
                        },
                        f,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
            except Exception:
                pass
        _ = tvm
        return ex, spec, cache_key_hash

    def _buf(self, storage_plan: StoragePlan, value_name: str) -> str:
        logical = storage_plan.value_to_buffer.get(value_name)
        if logical is None:
            raise KeyError(f"value not found in storage_plan: {value_name}")
        phys = storage_plan.to_physical(logical)
        if storage_plan.physical_buffers and phys not in storage_plan.physical_buffers:
            raise KeyError(f"physical buffer_id not found in storage_plan.physical_buffers: {phys} (value={value_name})")
        return phys

    def run_ibp_task(self, task, *, env: Dict[str, IntervalState], params: Dict[str, Any], storage_plan) -> None:
        """
        Execute a single INTERVAL_IBP task in-place on a shared env (BufferEnv: physical_id -> IntervalState).

        v0: accelerate `linear` via TVM Relax (kernel_style=relax/call_tir); other ops fall back to torch.
        """
        from ..ir.task import TaskKind  # local import to avoid cycles

        if task.kind != TaskKind.INTERVAL_IBP:
            raise NotImplementedError(f"TVMTaskExecutor only supports INTERVAL_IBP, got {task.kind}")
        if not isinstance(storage_plan, StoragePlan):
            storage_plan = storage_plan  # type: ignore[assignment]

        # Device anchor for param constants.
        device = None
        for v in env.values():
            device = v.lower.device
            break
        if device is None:
            for p in params.values():
                if torch.is_tensor(p):
                    device = p.device
                    break

        def get_interval(value_name: str) -> IntervalState:
            bid = self._buf(storage_plan, value_name)
            if bid in env:
                return env[bid]
            if value_name in params:
                t = params[value_name]
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t, device=device)
                s = IntervalState(lower=t, upper=t)
                env[bid] = s
                return s
            raise KeyError(f"missing value in env/params: {value_name}")

        def get_tensor(value_name: str) -> Any:
            if value_name in params:
                return params[value_name]
            raise KeyError(f"missing param tensor: {value_name}")

        target = self._select_target()
        import tvm
        from tvm.runtime import _tensor as rt
        from tvm import relax

        dev = tvm.cpu(0) if target == "llvm" else tvm.cuda(0)

        # PR#11A: try to run the whole task as a single Relax function (RELAX_OPS lowering).
        if self.options.enable_task_relax_ops and self.options.kernel_style == "relax":
            try:
                ex, spec, _cache_key = self._compile_interval_task_relax_ops(task, storage_plan=storage_plan, target=target)
                vm, fn = self._get_vm_callable(cache_key_hash=_cache_key, ex=ex, dev=dev, func_name=spec.func_name)
                args: List[Any] = []
                # Inputs: for each v, pass (v_l, v_u)
                for v in spec.input_values:
                    st = get_interval(v)
                    args.append(rt.tensor(st.lower.detach().cpu().numpy(), device=dev))
                    args.append(rt.tensor(st.upper.detach().cpu().numpy(), device=dev))
                # Params
                for p in spec.param_values:
                    t = get_tensor(p)
                    if not torch.is_tensor(t):
                        t = torch.as_tensor(t, device=device)
                    args.append(rt.tensor(t.detach().cpu().numpy(), device=dev))

                out = fn(*args)
                # Flattened outputs: [o0_l,o0_u,o1_l,o1_u,...]
                flat = list(out)
                for i, ov in enumerate(spec.output_values):
                    y_l_t = flat[2 * i]
                    y_u_t = flat[2 * i + 1]
                    y_l = torch.from_numpy(y_l_t.numpy()).to(device)
                    y_u = torch.from_numpy(y_u_t.numpy()).to(device)
                    env[self._buf(storage_plan, ov)] = IntervalState(lower=y_l, upper=y_u)  # type: ignore[assignment]
                return
            except Exception:
                # Fallback to per-op execution if the task lowering is not supported yet.
                pass

        def _run_linear(x: IntervalState, w: torch.Tensor, b: torch.Tensor) -> IntervalState:
            B, I = x.lower.shape
            O = w.shape[0]
            dtype = str(x.lower.dtype).replace("torch.", "")
            key = IntervalLinearKey(batch=int(B), in_features=int(I), out_features=int(O), dtype=dtype, target=target)
            ex = self._compile_interval_linear_executable(key)
            cache_key_hash = hashlib.sha256(
                repr((str(self.options.kernel_style), str(self.options.compile_cache_tag), key)).encode("utf-8")
            ).hexdigest()[:16]
            vm, fn = self._get_vm_callable(cache_key_hash=cache_key_hash, ex=ex, dev=dev, func_name="main")
            x_l_t = rt.tensor(x.lower.detach().cpu().numpy(), device=dev)
            x_u_t = rt.tensor(x.upper.detach().cpu().numpy(), device=dev)
            w_t = rt.tensor(w.detach().cpu().numpy(), device=dev)
            b_t = rt.tensor(b.detach().cpu().numpy(), device=dev)
            out = fn(x_l_t, x_u_t, w_t, b_t)
            y_l_t, y_u_t = out[0], out[1]
            y_l = torch.from_numpy(y_l_t.numpy()).to(x.lower.device)
            y_u = torch.from_numpy(y_u_t.numpy()).to(x.lower.device)
            return IntervalState(lower=y_l, upper=y_u)

        for op in task.ops:
            if not isinstance(op, TaskOp):
                raise TypeError(f"unexpected op type: {type(op)}")

            if op.op_type == "linear":
                x = get_interval(op.inputs[0])
                w = get_tensor(op.inputs[1])
                b = get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
                if not torch.is_tensor(w):
                    w = torch.as_tensor(w, device=x.lower.device)
                if b is None:
                    b = torch.zeros((w.shape[0],), dtype=w.dtype, device=w.device)
                if not torch.is_tensor(b):
                    b = torch.as_tensor(b, device=w.device)

                # Only accelerate 2D weight.
                if w.dim() == 2 and x.lower.dim() == 2 and self.options.kernel_style in ("relax", "call_tir"):
                    out = _run_linear(x, w, b)
                else:
                    out = self.domain.affine_transformer(x, w, b, op="linear")
                env[self._buf(storage_plan, op.outputs[0])] = out  # type: ignore[assignment]
                continue

            if op.op_type == "relu":
                x = get_interval(op.inputs[0])
                env[self._buf(storage_plan, op.outputs[0])] = self.domain.relu_transformer(x)  # type: ignore[assignment]
                continue

            if op.op_type in ("add", "mul"):
                a = get_interval(op.inputs[0])
                b = get_interval(op.inputs[1])
                env[self._buf(storage_plan, op.outputs[0])] = self.domain.elementwise_transformer([a, b], op=op.op_type)  # type: ignore[assignment]
                continue

            # Fallback: keep behavior consistent with PythonTaskExecutor by delegating to domain for known ops.
            raise NotImplementedError(f"unsupported op_type in TVMTaskExecutor.run_ibp_task: {op.op_type}")

    def run_ibp(
        self, module: BFTaskModule, input_spec: LinfInputSpec, *, output_value: Optional[str] = None
    ) -> IntervalState:
        module.validate()
        task = module.get_entry_task()
        if task.kind != TaskKind.INTERVAL_IBP:
            raise NotImplementedError(f"TVMTaskExecutor only supports INTERVAL_IBP, got {task.kind}")

        target = self._select_target()

        params: Dict[str, Any] = {}
        raw_params = module.bindings.get("params", {})
        if isinstance(raw_params, dict):
            params.update(raw_params)

        x0 = input_spec.center
        eps = float(input_spec.eps)
        env: Dict[str, IntervalState] = {
            input_spec.value_name: IntervalState(lower=x0 - eps, upper=x0 + eps)
        }

        def get_interval(value_name: str) -> IntervalState:
            if value_name in env:
                return env[value_name]
            if value_name in params:
                t = params[value_name]
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t, device=x0.device)
                return IntervalState(lower=t, upper=t)
            raise KeyError(f"missing value in env/params: {value_name}")

        def get_tensor(value_name: str) -> Any:
            if value_name in params:
                return params[value_name]
            raise KeyError(f"missing param tensor: {value_name}")

        # v0: only accelerate non-batched linear (2D weight). Other ops run on torch.
        import tvm
        from tvm.runtime import _tensor as rt
        from tvm import relax

        dev = tvm.cpu(0) if target == "llvm" else tvm.cuda(0)
        tvm_ops: List[str] = []
        fallback_ops: List[str] = []

        def _cache_info_dict(info: Any) -> Dict[str, int]:
            # functools._lru_cache_wrapper.cache_info() returns a namedtuple.
            return {
                "hits": int(getattr(info, "hits", 0)),
                "misses": int(getattr(info, "misses", 0)),
                "maxsize": int(getattr(info, "maxsize", 0) or 0),
                "currsize": int(getattr(info, "currsize", 0)),
            }

        def _as_int_tuple(value: Any, *, dim: int) -> List[int]:
            if isinstance(value, int):
                return [int(value)] * dim
            if isinstance(value, (list, tuple)):
                if len(value) == dim:
                    return [int(v) for v in value]
                if len(value) == 1:
                    return [int(value[0])] * dim
            raise ValueError(f"invalid int tuple: {value} (expected dim={dim})")

        def _run_relax_linear(key: IntervalLinearKey, x: IntervalState, w: torch.Tensor, b: torch.Tensor) -> IntervalState:
            ex = build_relax_interval_linear_vm_exec(key)
            vm = relax.VirtualMachine(ex, dev)
            x_l_t = rt.tensor(x.lower.detach().cpu().numpy(), device=dev)
            x_u_t = rt.tensor(x.upper.detach().cpu().numpy(), device=dev)
            w_t = rt.tensor(w.detach().cpu().numpy(), device=dev)
            b_t = rt.tensor(b.detach().cpu().numpy(), device=dev)
            out = vm["main"](x_l_t, x_u_t, w_t, b_t)
            y_l_t, y_u_t = out[0], out[1]
            y_l = torch.from_numpy(y_l_t.numpy()).to(x0.device)
            y_u = torch.from_numpy(y_u_t.numpy()).to(x0.device)
            return IntervalState(lower=y_l, upper=y_u)

        def _run_relax_conv2d(key: IntervalConv2dKey, x: IntervalState, w: torch.Tensor, b: torch.Tensor) -> IntervalState:
            ex = build_relax_interval_conv2d_vm_exec(key)
            vm = relax.VirtualMachine(ex, dev)
            x_l_t = rt.tensor(x.lower.detach().cpu().numpy(), device=dev)
            x_u_t = rt.tensor(x.upper.detach().cpu().numpy(), device=dev)
            w_t = rt.tensor(w.detach().cpu().numpy(), device=dev)
            b_t = rt.tensor(b.detach().cpu().numpy(), device=dev)
            out = vm["main"](x_l_t, x_u_t, w_t, b_t)
            y_l_t, y_u_t = out[0], out[1]
            y_l = torch.from_numpy(y_l_t.numpy()).to(x0.device)
            y_u = torch.from_numpy(y_u_t.numpy()).to(x0.device)
            return IntervalState(lower=y_l, upper=y_u)

        for op in task.ops:
            if op.op_type == "linear":
                x = get_interval(op.inputs[0])
                w = get_tensor(op.inputs[1])
                b = get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
                if not torch.is_tensor(w):
                    w = torch.as_tensor(w)
                if b is None:
                    b = torch.zeros((w.shape[0],), dtype=w.dtype)
                if not torch.is_tensor(b):
                    b = torch.as_tensor(b)

                if w.dim() != 2:
                    # fallback for batched/spec-fused linear (handled by PythonTaskExecutor today)
                    fallback_ops.append("linear")
                    out = self.domain.affine_transformer(x, w, b, op="linear")
                    env[op.outputs[0]] = out  # type: ignore[assignment]
                    continue

                B, I = x.lower.shape
                O = w.shape[0]
                dtype = str(x.lower.dtype).replace("torch.", "")
                key = IntervalLinearKey(batch=int(B), in_features=int(I), out_features=int(O), dtype=dtype, target=target)
                if self.options.kernel_style == "te":
                    func = build_interval_linear_module(key)
                    x_l_t = rt.tensor(x.lower.detach().cpu().numpy(), device=dev)
                    x_u_t = rt.tensor(x.upper.detach().cpu().numpy(), device=dev)
                    w_t = rt.tensor(w.detach().cpu().numpy(), device=dev)
                    b_t = rt.tensor(b.detach().cpu().numpy(), device=dev)
                    y_l_t = rt.empty((B, O), dtype=dtype, device=dev)
                    y_u_t = rt.empty((B, O), dtype=dtype, device=dev)
                    func(x_l_t, x_u_t, w_t, b_t, y_l_t, y_u_t)
                    y_l = torch.from_numpy(y_l_t.numpy()).to(x0.device)
                    y_u = torch.from_numpy(y_u_t.numpy()).to(x0.device)
                    env[op.outputs[0]] = IntervalState(lower=y_l, upper=y_u)
                else:
                    env[op.outputs[0]] = _run_relax_linear(key, x, w, b)  # type: ignore[assignment]
                tvm_ops.append("linear")
                continue

            if op.op_type == "conv2d":
                x = get_interval(op.inputs[0])
                w = get_tensor(op.inputs[1])
                b = get_tensor(op.inputs[2]) if len(op.inputs) >= 3 else None
                if not torch.is_tensor(w):
                    w = torch.as_tensor(w)
                if b is None:
                    b = torch.zeros((w.shape[0],), dtype=w.dtype)
                if not torch.is_tensor(b):
                    b = torch.as_tensor(b)

                stride = _as_int_tuple(op.attrs.get("stride", 1), dim=2)
                padding = _as_int_tuple(op.attrs.get("padding", 0), dim=2)
                dilation = _as_int_tuple(op.attrs.get("dilation", 1), dim=2)
                groups = int(op.attrs.get("groups", 1))

                if x.lower.dim() != 4 or w.dim() != 4 or groups != 1:
                    fallback_ops.append("conv2d")
                    out = self.domain.affine_transformer(
                        x,
                        w,
                        b,
                        op="conv2d",
                        stride=tuple(stride),
                        padding=tuple(padding),
                        dilation=tuple(dilation),
                        groups=groups,
                    )
                    env[op.outputs[0]] = out  # type: ignore[assignment]
                    continue

                N, CI, H, W = x.lower.shape
                CO, CI_w, KH, KW = w.shape
                if CI_w != CI:
                    fallback_ops.append("conv2d")
                    out = self.domain.affine_transformer(
                        x,
                        w,
                        b,
                        op="conv2d",
                        stride=tuple(stride),
                        padding=tuple(padding),
                        dilation=tuple(dilation),
                        groups=groups,
                    )
                    env[op.outputs[0]] = out  # type: ignore[assignment]
                    continue

                dtype = str(x.lower.dtype).replace("torch.", "")
                key = IntervalConv2dKey(
                    batch=int(N),
                    in_channels=int(CI),
                    in_h=int(H),
                    in_w=int(W),
                    out_channels=int(CO),
                    k_h=int(KH),
                    k_w=int(KW),
                    stride_h=int(stride[0]),
                    stride_w=int(stride[1]),
                    pad_h=int(padding[0]),
                    pad_w=int(padding[1]),
                    dilation_h=int(dilation[0]),
                    dilation_w=int(dilation[1]),
                    groups=int(groups),
                    dtype=dtype,
                    target=target,
                )
                if self.options.kernel_style == "te":
                    func = build_interval_conv2d_module(key)
                    oh = (int(H) + 2 * int(padding[0]) - int(dilation[0]) * (int(KH) - 1) - 1) // int(stride[0]) + 1
                    ow = (int(W) + 2 * int(padding[1]) - int(dilation[1]) * (int(KW) - 1) - 1) // int(stride[1]) + 1
                    x_l_t = rt.tensor(x.lower.detach().cpu().numpy(), device=dev)
                    x_u_t = rt.tensor(x.upper.detach().cpu().numpy(), device=dev)
                    w_t = rt.tensor(w.detach().cpu().numpy(), device=dev)
                    b_t = rt.tensor(b.detach().cpu().numpy(), device=dev)
                    y_l_t = rt.empty((int(N), int(CO), int(oh), int(ow)), dtype=dtype, device=dev)
                    y_u_t = rt.empty((int(N), int(CO), int(oh), int(ow)), dtype=dtype, device=dev)
                    func(x_l_t, x_u_t, w_t, b_t, y_l_t, y_u_t)
                    y_l = torch.from_numpy(y_l_t.numpy()).to(x0.device)
                    y_u = torch.from_numpy(y_u_t.numpy()).to(x0.device)
                    env[op.outputs[0]] = IntervalState(lower=y_l, upper=y_u)
                else:
                    env[op.outputs[0]] = _run_relax_conv2d(key, x, w, b)  # type: ignore[assignment]
                tvm_ops.append("conv2d")
                continue

            if op.op_type == "relu":
                x = get_interval(op.inputs[0])
                env[op.outputs[0]] = self.domain.relu_transformer(x)  # type: ignore[assignment]
                fallback_ops.append("relu")
                continue

            if op.op_type == "add":
                a = get_interval(op.inputs[0])
                b = get_interval(op.inputs[1])
                env[op.outputs[0]] = self.domain.elementwise_transformer([a, b], op="add")  # type: ignore[assignment]
                fallback_ops.append("add")
                continue

            # v0: fallback to torch semantics for remaining ops
            if op.op_type == "flatten":
                x = get_interval(op.inputs[0])
                start_dim = int(op.attrs.get("start_dim", 0))
                end_dim = int(op.attrs.get("end_dim", -1))
                env[op.outputs[0]] = IntervalState(
                    lower=torch.flatten(x.lower, start_dim=start_dim, end_dim=end_dim),
                    upper=torch.flatten(x.upper, start_dim=start_dim, end_dim=end_dim),
                )
                fallback_ops.append("flatten")
                continue

            if op.op_type == "reshape":
                x = get_interval(op.inputs[0])
                shape = op.attrs.get("shape")
                if shape is None:
                    env[op.outputs[0]] = x
                    fallback_ops.append("reshape")
                    continue
                env[op.outputs[0]] = IntervalState(lower=x.lower.reshape(shape), upper=x.upper.reshape(shape))
                fallback_ops.append("reshape")
                continue

            if op.op_type in ("permute", "transpose"):
                x = get_interval(op.inputs[0])
                dims = op.attrs.get("dims")
                if not isinstance(dims, (list, tuple)):
                    raise ValueError(f"transpose missing dims for op '{op.name}': {dims}")
                dims = [int(d) for d in dims]
                env[op.outputs[0]] = IntervalState(lower=x.lower.permute(*dims), upper=x.upper.permute(*dims))
                fallback_ops.append("permute")
                continue

            if op.op_type == "spec_linear":
                # v0: keep spec_linear on torch (not performance critical compared to affine core).
                logits = get_interval(op.inputs[0])
                C = get_tensor(op.inputs[1])
                if not torch.is_tensor(C):
                    C = torch.as_tensor(C, device=x0.device)
                C_pos = torch.clamp(C, min=0.0)
                C_neg = torch.clamp(C, max=0.0)
                l = logits.lower.unsqueeze(1)
                u = logits.upper.unsqueeze(1)
                lb = (C_pos * l + C_neg * u).sum(dim=-1)
                ub = (C_pos * u + C_neg * l).sum(dim=-1)
                env[op.outputs[0]] = IntervalState(lower=lb, upper=ub)
                fallback_ops.append("spec_linear")
                continue

            raise NotImplementedError(f"unsupported op_type in TVMTaskExecutor: {op.op_type}")

        if output_value is None:
            if len(task.output_values) != 1:
                raise ValueError(f"task has {len(task.output_values)} outputs; specify output_value explicitly")
            output_value = task.output_values[0]
        self.last_stats = TVMRunStats(
            tvm_ops=tvm_ops,
            fallback_ops=fallback_ops,
            linear_kernel_cache=_cache_info_dict(
                (build_interval_linear_module if self.options.kernel_style == "te" else build_relax_interval_linear_vm_exec).cache_info()
            ),
            conv2d_kernel_cache=_cache_info_dict(
                (build_interval_conv2d_module if self.options.kernel_style == "te" else build_relax_interval_conv2d_vm_exec).cache_info()
            ),
            kernel_style=self.options.kernel_style,
        )
        return get_interval(output_value)
