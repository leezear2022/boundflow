---
status: superseded-by-full-stack-overlay
updated: 2026-08-06T09:06:51Z
type: changelog
topic: boundflow
slug: gpu-compiler-acceleration-research-v1
stage: s01
revision: v1.2-full-stack-overlay
---

# BoundFlow GPU 编译器加速调研计划变更记录

## 2026-08-06：v1.2 full-stack overlay

- 保留 NRIR49A/G1 的全部数据、artifact 和 hash，但将 verdict 作用域精确收窄为
  `VALIDATED-NO-GO(selected-CROWN-only incremental optimization)`；
- 明确 `1.0764x` 只是 selected-CROWN deletion-only Amdahl 上限，不是算子→IR/图→
  Plan/Schedule→JIT/cache→runtime→allocator/memory 的 BoundFlow 全栈上限；
- 旧 G2—G4 保留为历史预注册与 gated 路线，不再作为当前执行指令；
- 不改写冻结 artifact 中的 `next_route=gpu-winner-reselection`，但将其标记为已被
  [full-stack plan](BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md)
  取代的历史机器输出；
- 唯一执行顺序改为：FSG0 schema/critical-path/replay合同→FSG1 official αβ-CROWN B0 full-stack
  baseline；FSG0现已关闭，当前下一步为FSG1，不再重新寻找一个单点winner；
- 本轮只修正路线和 claim 边界，不实现优化，不产生任何 BoundFlow GPU 性能主张。

## Summary

- 新增一份 research-only 的 GPU 编译器加速诊断与分阶段计划；
- 不修改 Python/C++/TIR/runtime，不产生新的性能 claim；
- 将 CUDA/TIR、流程融合、物理内存、多分支并行和 JIT 收敛为一条有依赖和 kill gate 的路线；
- 明确下一步只做 G0 环境/证据恢复，不直接进入 kernel 实现。
- 2026-08-06根据用户评审升级为v1.1：把G8可达性、solveability和physical-memory admission前移到
  G0/G1，仍不启动TIR实现。
- G0 pre-reboot execution 已关闭 competitor env 与 solveability blocker；当前仅等待 firmware delayed
  apply 后的重启/CUDA smoke，详见独立 G0 admission 文档。

## Changes

- 盘点当前分支、HEAD、三个 submodule、conda/PyTorch/TVM/CUDA 状态；
- 复核现有 Bound/Plan/Task/Schedule IR、fused executor/cache、logical storage、batch manager 与
  reference Schedule executor的ownership缺口；
- 定位 `native_intermediate_refinement._run_selected_crown` 的 dense one-hot、per-chunk allocation、
  repeated backward和未传 fused executor问题；
- 汇总 PR-12I/J/K/L、PR-13D、IR-5、NRIR43/46/47/48 的正负证据；
- 用 Amdahl 粗算明确 micro-kernel `40x` 不能自动成为 whole-query `40x`；
- 将用户报告的 BoundConv `40x` 标为 `USER-REPORTED`，列入 G0 复现而非仓库 claim；
- 定义 G0—G8：GPU恢复与公平复现、GPU归因、现有fused wiring、SelectedObjectiveIR/TIR、
  frozen alpha/beta、physical arena、ragged batch/stream、条件JIT/Graph、same-solver E2E；
- 新增 claim registry，以独立 ID 区分当前事实、历史测量、用户线索、研究假设和建议门禁；
- 识别当前 PyTorch `2.12.1` 与 vendored auto_LiRPA 声明的 `<2.9.0` 约束，要求独立锁定的
  competitor env/container与跨环境同GPU/工件/timing合同；
- 将用户40x证据与GPU基础设施拆为两个verdict：源码缺失时禁止40x claim，但不阻止独立G1 profiling；
- 补齐ResNet backward中的residual add/fanout、concat、flatten/reshape边界分类与legality/replay gate，
  防止新region再次退化为未说明边界的ReLU+单affine局部融合；
- 将auto_LiRPA已有memory-efficient GPU BoundConv/ConvTranspose backward列为source-level强baseline，
  新TIR不得只对比eager/unfused；
- 给出 benchmark matrix、artifact/replay合同、预注册go/no-go门槛、PR切分和外部模型审计模板；
- 对照 auto_LiRPA、TVM、PyTorch compilation/cache、CUDA streams/graphs 的官方边界，限定新颖性。
- 新增精确Amdahl反解`r=s/(s+1/T-1)`，分别绑定queue `1.20x`与complete-query `1.15x`；
  `required>10x`或不可达时不开latency G3；
- benchmark矩阵硬性加入至少一个control/candidate均非unknown的公开held-out workload；
- G1新增`B80_alloc/B80_reserved/B_OOM/max-valid-batch`，提前判定目标GPU上physical-memory path是否真实可达；
- G8主对照冻结为同一alpha-beta-CROWN host solver内，RVIR exact-call合同下original batched executor
  对BoundFlow replacement executor；非对称跨env计时不得作为headline；
- 增加本机GPU恢复timebox、备用GPU最低规格与建议资源上限，以及逐op frontend coverage admission；
- G2资格审查timebox为2 engineer-days/1 PR；G1 chunk sweep明确只读、不改production默认；
- Planner claim缩为shape/cache/memory/reuse驱动的GPU-context selector，不复活IR-5 broad global planner；
- 修正外部审计M-1：PR-12J compile phase改为`0.324/0.480/1.299 s`并链接审计报告。

## Validation

- tracked diff的 `git diff --check` PASS；两个untracked新文档分别以
  `git diff --no-index --check /dev/null <file>` 等价检查，均无whitespace error；
- 15 个仓库内相对Markdown链接全部存在；G0—G8九个阶段、12个claim ID和新增的Amdahl、
  solveability、memory、RVIR、timebox、GPU-context术语全部可定位；
- `awk`独立重算Amdahl示例为`3x/6x/38.8845x/4.21021x`，与正文一致；PR-12J原文
  compile phase=`323.67/480.00/1299.12 ms`，M-1修正成立；
- PR-12I/J/K/L、PR-13D、IR-5、NRIR48数字、Amdahl推导和关键源码入口均已逐项复核；
- 独立只读文档审计初判 `approve-with-minor`（0 blocker / 0 major / 5 minor），5项minor已在本版
  全部修正；
- DocOps scaffold events=`ev005726/ev005727`，change=`ev005813`，validation=`ev005819`；
  v1.1 review change=`ev005858`、validation=`ev005859`；`dol lint --soft`为`ok=true, miss=[]`；
- docs-only，未运行代码测试，也未进行 GPU benchmark；本会话当前无法访问 NVIDIA driver/GPU。

## Decisions

- 最高优先级不是泛化继续调旧 plain-CROWN TIR，而是 production selected-CROWN 的
  verification-aware GPU compilation；
- G2先做 split/relaxation/kernel公式 legality qualification；现有runtime硬拒split/alpha/beta，不能只传
  fused参数或改capability名称；
- G5将host-sync速度准入与peak/OOM memory准入拆开，避免allocator时间占比错误阻断arena路线；
- packed sibling batching先于multi-stream；physical arena先于CUDA Graph；AOT/cache family先于JIT；
- 新路线不覆盖 PR-12L、IR-5、NRIR43/46/47 的历史 NO-GO；
- 任何性能claim都必须传播到region、child、queue和complete-query，并比较公平batched baseline；最终
  latency path冻结为queue geomean `>=1.20x` 且complete-query `>=1.15x`，另设严格memory path；
- G1先反解达到两级latency gate所需的region speedup；不可达或`>10x`直接关闭latency工程投入；
- GPU-context selector/C2升级必须通过held-out p90 Oracle regret`<=1.20x`，否则降级为narrow
  backend/memory claim；IR-5 broad global planner保持NO-GO；
- 40x未能公平复现只降级 `U-40X-01`，不阻塞独立G1；GPU opportunity未通过时停止selected-CROWN
  GPU路线；global planner仍不优于fixed plan时降级C2/adaptive claim。

## Follow-Ups

1. FSG0 full-stack schema、critical-path/interaction聚合器、artifact replay与确定性测试已完成；
2. 当前进入FSG1，在official αβ-CROWN control侧生成B0 full-stack baseline；
3. B0只用于分层归因和后续same-solver A/B的公平分母，不宣称BoundFlow speedup；
4. RVIR replacement correctness、B0/B1/B2 paired timing与B3—B7累计消融均按full-stack plan
   的后续门禁执行；不恢复旧 G2—G4 单点路线。

## Links

- plan: [BoundFlow GPU 编译器加速诊断与执行计划 v1](BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md)
- roadmap: [ASPLOS master plan](boundflow_asplos_master_plan_2026_07_12.md)
- external audit: [GPU 编译计划 v1 外部审计](external_audit_gpu_compiler_plan_v1_2026_08_05.md)
- G0 execution: [NRIR49 G0 GPU opportunity admission](BOUNDFLOW_NRIR49_G0_GPU_OPPORTUNITY_ADMISSION_V1_PLAN_2026_08_06.md)
- current route: [Full-stack GPU baseline and attribution](BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md)
