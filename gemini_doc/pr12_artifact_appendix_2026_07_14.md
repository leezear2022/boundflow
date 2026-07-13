# PR-12 reduced artifact appendix

## Scope

该工件验证 static FP32 CUDA plain-CROWN 的 fused Linear/Conv mechanism、公平 baseline、compile
amortization、CUPTI activity 与 compile-aware 多预算 Planner。它不验证真实 BaB、α/β/grad、
VNN-COMP full scale，也不提供硬件 performance counter 结论。

## Requirements

- 激活 Conda 环境：`conda activate boundflow`；
- CUDA GPU、PyTorch 2.12.1+cu132、仓库构建的 TVM/tvm-ffi；
- `source env.sh`；
- profiler 部分可用 CUPTI activity；ncu hardware counter 在当前主机预期返回
  `ERR_NVGPUCTRPERM`，这不是 smoke failure。

## Reduced verification

```bash
conda activate boundflow
source env.sh

pytest -q \
  tests/test_phase7a_pr12i_baseline_runner.py \
  tests/test_phase7a_pr12j_amortization_runner.py \
  tests/test_phase7a_pr12k_cupti_profiler.py \
  tests/test_phase7a_pr12m_compile_aware_planner.py \
  tests/test_phase7a_pr12m_compile_aware_replay.py \
  tests/test_phase7a_pr12m_postprocess.py
```

预期：所有专项测试通过；仅 profiler 可能打印已知的 PyTorch event-cycle warning。

## PR-12M evidence workflow

完整路径和正式参数见
`gemini_doc/change_2026-07-14_pr12m_compile_aware_planner.md`。顺序必须是：

```text
start_phase7a_pr12_m.py
  → calibration baseline
  → fit_phase7a_pr12m_compile_aware.py
  → verify final_heldout_consumed=false + freeze model SHA
  → final-heldout baseline（一次）
  → replay_phase7a_pr12m_compile_aware.py
  → postprocess_phase7a_pr12m_compile_aware.py
```

禁止先运行 final-heldout 再调模型。正式 expected outputs：

```text
split SHA:          1f79962d7d6325fbfbf6b0d9f63fef93e4a5e9866c840b75bda7374ecf2c5f83
calibration SHA:    ae78a7147fb51d25737ce34dd95cf475b77a450b317af045af8202000f45bf59
model SHA:          dc56c58b83ea355097ff14fe42e48599d16b3ed3e391c7c30f3febf7b2dcfa59
final-heldout SHA:  54dd6467201eec3e1e4522452ca88bb82bdf85408af735d3832586b1a2b2d03d
planner JSONL SHA:  cb093ed53ce3d1b76aa0a18e3e38bced2fe4809f7ecce05a079bbdc91be429a9
summary SHA:        bd84fa171752fcefd6620f05329ddf31edf5a161ec915173714ca39a870b3268
```

预期 summary：75 rows；72 feasible opportunities；72 selected feasible；0 unsafe；feasible
median/p90/max regret 1.000/1.000/1.016×；3 个 16 MiB no-feasible rows。

## Claims map

- mechanism/correctness：`C1-E4`、`C2-E6`；
- fair baseline/memory：`C2-E13`、`C2-E14`；
- compile/cache limitation：`C3-M1`、`C3-E1`、`C3-L1`；
- profiler/stop decision：`C2-E15`、`C2-E16`、`C2-D1/D2`；
- compile-aware held-out：`C2-E17`–`C2-E20`、`C2-L9`。

动态映射以 `gemini_doc/asplos_claims_map.md` 为准。
