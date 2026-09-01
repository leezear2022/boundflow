---
status: executed-invalid
updated: 2026-08-26T00:25:00+08:00
type: plan
topic: boundflow
slug: mr7-launch-materialization-attribution
stage: s01
---

# MR7 Launch / Materialization Attribution 预注册计划

## 1. 目标

在不改变solver、TIR数学、schedule或production default的前提下，把MR6 diagnostic bridge剩余时间拆成
互斥、可闭合的物理类别，判断下一变量应该是：

1. cross-site/step launch与FFI amortization；
2. layout/materialization与buffer reuse；
3. site-specific TIR schedule/autotuning；
4. 或者当前路线整体NO-GO。

MR7同时是全编译验证器运行时路线的`FCR-0`入口门。完整目标、编译边界和分阶段迁移见
`BOUNDFLOW_FULLY_COMPILED_VERIFIER_RUNTIME_V1_ARCHITECTURE_2026_08_25.md`。MR7只负责建立物理账本，
但其开放的MR7-A/MR7-C后续必须以多算子compiled region、统一execution graph或arena为目标，不能
退化为继续堆叠逐算子的PyTorch↔TVM wrapper。

## 2. Scope与目标冻结

- workload、10/9、C2→C1→C0、module和diagnostic guard policy继承MR6；
- 主baseline仍是native provider；candidate测量对象是MR6 diagnostic，不把unsafe结果当production；
- parity所需candidate region speedup=`1.107412x`；
- provider上`1.15x`研究目标所需candidate region speedup=`1.273523x`；
- graph/query/queue share不得代入本outer scope；
- profiler只做归因，headline性能仍来自独立unprofiled host/event control。

## 3. 互斥类别

每个site/evaluation按NVTX ordinal绑定以下类别：

1. `admission_handoff`：typed structural checks、α重建、beta census、lineage；
2. `layout_materialization`：permute/transpose/contiguous、zero/empty tensor创建；
3. `ffi_dlpack_stream`：DLPack view、pointer、TVM-FFI stream/device envelope；
4. `forward_device_kernel`：30个forward kernel的device时间，按C0/C1/C2分开；
5. `backward_device_kernel`：27个backward kernel的device时间，按C0/C1/C2分开；
6. `post_output_guard`：输出layout和60次finite guard；
7. `optimizer_and_residual`：outer中未属于上述region的provider-owned optimizer/residual。

CUDA kernel必须由显式marker correlation归属；temporal fallback只能用于非headline residual并单列。
CUPTI↔host/NVTX calibration残差超`max(5us,2%)`则本run不得形成share。

## 4. Protocol

- 3 fresh counterbalanced control/profile pairs；control不启profiler，profile启NVTX+CUPTI；
- profile/control outer event ratio必须`<=1.10`，否则归因INVALID；
- 3/3 semantic、30/27、module/cache/stream exact；
- 所有类别host span闭合误差`<=2%`，device kernel envelope闭合误差`<=2%`；
- 报告每类absolute ns、outer share、三site分布、MAD和跨run range；
- raw-first、独立replay、fully re-signed category/correlation/calibration tamper。

## 5. 冻结路由

- 若`ffi_dlpack_stream + layout_materialization + post_output_guard` median outer share `>=15%`，且
  absolute median `>=15 ms`：开放MR7-A persistent buffer + batched FFI/launch plan；
- 否则若forward/backward device kernel合计share `>=50%`，且至少同一site在3/3 run为最慢：开放
  MR7-B per-site schedule sweep，固定64/128/256，不新增事后schedule；
- 若launch envelope本身`>=15%`但单site kernel不dominant：开放MR7-C cross-site或cross-step
  schedule/graph amortization feasibility，先做ABI/correctness，不直接优化；
- 三项均不满足：当前MR5/R3 production Conv replacement路线总NO-GO，不再沿本路径加工程量。

上述“总NO-GO”只关闭当前逐站点production Conv replacement，不关闭FCR路线中optimizer、branch/queue、
memory planning或其他由独立归因支持的compiled region。

无论哪支开放，都必须先用Amdahl公式验证该share在`1.107412x` parity目标下数学可达；required region
speedup `>10x`则直接NO-GO。

## 6. 禁止项

- 不复活MR6-B guard fusion；
- 不复活B4-C2 dense cross-layer retention；
- 不用独立CIBC-IBP `2.4563x`替代same-solver数据；
- 不在归因阶段修改schedule、allocator、kernel、solver或阈值；
- 不形成complete-query、queue、competitor或ASPLOS-ready claim。

## 7. 执行结果注解

6 fresh/3 pair已执行。correctness、30/27、host closure、device envelope和11/11 tamper通过，但
profile/control ratio=`1.239399/1.039553/1.096733`，第1组超过`1.10`，故正式状态为
`INVALID_MR7_ATTRIBUTION`，MR7-A/B/C均未开放。诊断host boundary median=`25.891 ms/19.818%`，
只用于预注册MR7-R，不是opportunity claim。见
`BOUNDFLOW_MR7_LAUNCH_MATERIALIZATION_FORMAL_INVALID_CLOSURE_2026_08_26.md`。
