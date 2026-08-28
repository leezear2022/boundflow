---
status: implementation-in-progress
date: 2026-08-28
type: changelog
topic: boundflow
slug: asplos27-s1-canonical-cibc-pipeline-change
performance-claimed: false
---

# ASPLOS’27 S1 canonical CIBC pipeline 修改记录

## 本批改动

- 扩展 `relax_interval_task_ops.py`：按 exact op name 绑定 CIBC schedule，6 个 Conv 通过标准
  `Relax call_tir` 进入同一 17-op function；缺失、重复、未知或非法 schedule 均拒绝；
- 为 Relax 增加双 DPS output CIBC PrimFunc，消除原 combined-output 后的 12 个 slice/materialize kernel；
- Linear 的 transpose 保持为每个 matmul 私有表达式，使 TVM cuBLAS partition 能吸收 transpose，消除独立
  transpose kernel；
- 新增 `PreparedS1CIBCProgramV1` 与 `PreparedS1CIBCCUDAGraphV1`：compile/prepare/run、source→Relax→device
  identity、persistent DLPack views、input/param version guard、execution receipt；
- CUDA Graph 输出直接绑定 capture-stable VM result，warm replay不构造 DLPack、不做额外output copy；
- `install_dev.sh` 固定 `USE_CUBLAS=ON`，verify 检查 TVM cuBLAS codegen/runtime symbol；
- 新增正负向测试、fresh worker、root artifact replay 与 8 类 fully outer-resigned tamper probe。

## 实现期诊断

以下是实现 smoke，不是 formal claim：

1. 初始未捕获 VM约 `0.281 ms`，只为 PyTorch baseline 的 `0.615x`；
2. CUDA Graph 后约 `0.204 ms`，仍慢于 PyTorch；
3. 发现 TVM 构建 `USE_CUBLAS=OFF`，通用 DLight Linear 是主瓶颈；
4. 打开 cuBLAS 后约 `0.098 ms`，但仅保住 direct-CIBC 的约 `0.72x`；
5. 去掉 shared transpose materialization 与 CIBC slice kernel 后约 `0.070 ms`；
6. 第一次 graph output copy 使 upper 出现约 `0.46` 偏差；逐层 debug 证明 VM final pair本身正确，偏差来自
   captured copy；改为prepare-time stable output binding后恢复至 lower/upper最大约`1.83e-4/2.44e-4`。

最终 formal 数字只能来自冻结的 six-fresh artifact，不能引用上述 smoke。

## 当前验证

- targeted：`9 passed`；
- 单 fresh worker smoke：PyTorch/direct/pipeline约
  `0.17293/0.07028/0.07022 ms`，pipeline/PyTorch约`2.4626x`，pipeline/direct约`1.0009x`；
- formal six-fresh、全量测试、静态检查与 DocOps closure：待执行。
