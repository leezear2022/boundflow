---
status: fsg2-initial-validated-b2-and-downstream-gated
updated: 2026-08-06T13:54:29Z
type: changelog
topic: boundflow
slug: BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1
stage: s01
---

# BoundFlow Full-Stack GPU Baseline and Attribution v1 Changelog

## FSG4/B4 Cumulative Fusion Preregistration

- B3为直接基线、B0为累计对照；
- 从raw冻结14次lower-only CROWN的Amdahl share与required-speedup公式；
- B4依次拆为B4-0 attribution、B4-A terminal export fusion、B4-B differentiable lower-only TIR、
  B4-C cumulative coverage与B4-D formal timing；
- 当前`PREREGISTERED-NOT-IMPLEMENTED`，只执行B4-0，B5—B7关闭。

## FSG4/B3 External Audit Closure

- Round 2从36个raw worker独立重算44项检查，AC1—AC7全PASS，无blocker/major/minor；
- exchange已`closed/approved`，`VALIDATED-REDUCED-B3`正式关闭；
- 只开放以B3为直接增量对照的B4 operator/cross-stage fusion，B5—B7及最终system gate关闭。

## FSG4/B3 Formal Timing Internal Closure

- source `36e9069`完成B0/B2/B3六全排列36-process正式artifact；correctness、environment、activation、
  measurement、root replay与10/10 tamper通过；
- B2/B3 core/query=`1.071617x/1.006623x`，B0/B3 query=`0.910001x`，分类=
  `VALIDATED-REDUCED-B3`，没有B0 parity或全栈speedup claim；
- frozen=`6 passed`、targeted=`114 passed`、full=`1314 passed, 3 skipped`；下一动作为external audit，
  通过后只开放B4 cumulative candidate。

## FSG2 Closure and FSG3—FSG5 Dependency Gate

- RVIR-v3 no-original-callback initial-CROWN replacement在冻结ResNet上lower diff=`7.152557e-7`、
  sign=`9/9`，正式artifact replay通过；
- production inventory捕获`12 initial + 1 alpha + 11 beta/split` calls；alpha state为嵌套
  start-node keyed tensors，beta/split前后均无显式beta tensor ownership；
- FSG2以initial-only `VALIDATED-REDUCED`关闭，完整B2 `NO-GO/not admitted`；FSG3—FSG5因依赖
  B2而未运行，不形成性能claim，也不宣称B3—B7各层潜力被证伪；
- inventory v2 summary/manifest=`37f6dbcd…6544`/`e8548a25…ff06`；全量
  `1107 passed, 3 skipped`。

## FSG1 Formal Closure

- 两workload各5 fresh AB/BA pairs，10/10 result/visited-domain exact与attribution closure通过；
- ResNet/MNIST perturbation median=`1.026200/1.001089<=1.05`；ResNet profile固定234 calls、
  visited domains=`[6064]`，MNIST 1 call并自然verified；
- artifact replay exact；summary/manifest hash=`1e5f2946…7d92`/`c9496d27…d1e`；
- FSG1只关闭B0 measurement denominator，`performance_claimed=false`；当时下一步为FSG2，现已由
  本文顶部的FSG2 closure取代。

## FSG1 Fixed-Iteration Protocol Correction

- 首次正式 60 秒 ResNet control/profile 因 timeout 边界产生约 `150022/150018` visited-domain
  漂移，exact work gate 不成立，执行方中止该轮且不采信其数字；
- 改用 official `bab/max_iterations=16` 固定求解前缀，wall timeout 仅作 fail-closed 保险丝；
- fixed-16 首次 smoke 暴露 auto batch 的 allocator-dependent 漂移（`18944/18954`），因此正式协议
  关闭 auto enlargement并固定/重置 seed；batch64 单pair扰动=`1.060620>1.05`，故在任何 candidate
  前冻结batch=256；这些 diagnostics 均不进入性能证据；
- batch256 diagnostic仍为`1.075754`，进一步将observer phase识别从构造完整`inspect.stack()`
  改为只读`f_back`遍历；字段、CUDA event与solver配置均不变；
- 轻量observer后续diagnostic ratio=`1.032419<=1.05`，result exact、visited domains均`[6064]`、
  profile calls=234；只作为measurement admission，不进入正式performance evidence；
- 该预算进入 raw protocol、control/profile exact comparison 与 replay，不构成 complete-query 或
  BoundFlow performance claim。
- 定向`10 passed`、全量`1089 passed, 3 skipped`、Black、mypy、Pylint 10.00/10和
  `git diff --check`通过；正式artifact须绑定本修正的clean code revision。

## FSG1 Runner Preparation

- 新增official αβ-CROWN B0 control/profile typed worker合同与full-stack重建器；
- 新增独立Python 3.11/Torch 2.11 CUDA worker、交替AB/BA fresh-process编排和raw-first artifact/replay；
- compute-bound observer记录嵌套host/CUDA event、solver phase、stream、allocator counters并可逆恢复；
- fresh isolated VNNLIB副本避免`.compiled`缓存污染pair中的第二个worker；
- 真实`mnistfc:2` smoke result exact，ratio=`1.014834<=1.05`，捕获1个initial-CROWN call；
- 定向`10 passed`、全量`1089 passed, 3 skipped`，三个新文件mypy clean、Pylint 10.00/10；
  正式五轮结果尚未运行，
  `performance_claimed=false`。

## Summary

- 纠正NRIR49A结论传播层级：保留selected-CROWN-only NO-GO，不外推BoundFlow全栈GPU上限；
- 以official αβ-CROWN same-solver、full-stack hierarchical attribution和B0—B7累计/leave-one-out
  消融替代“寻找下一个单点winner”路线；
- 本轮只关闭FSG0合同/schema切片，没有新增speedup或production优化；当前下一步为FSG1。

## Changes

- 用DocOps创建独立plan/changelog，作为新的唯一当前路线入口；
- 冻结operator、graph/IR、Plan/Schedule、backend compile/JIT、runtime scheduling、memory allocator、
  solver/adapter九层与solver phase/resource/cache四轴schema；
- 冻结host wall、GPU union、GPU sum、critical path和exclusive critical-path分离口径；
- 冻结B0 original→B1 typed transport→B2 replacement→B3 IR/graph→B4 fusion→B5 JIT→B6 runtime→
  B7 arena/reuse累计链；
- 明确当前RVIR只是original callable exactly-once transport，PR13C也不是official host solver；
- 将combined environment或对称RPC设为same-solver headline强制前提；
- 预注册FSG0—FSG5、correctness/measurement/system gate、13文件artifact与raw replay。
- 新增typed full-stack attribution合同：九个功能层加一个residual哨兵、九个功能phase加
  `setup/unclassified`两个哨兵、host/CUDA/runtime/memory/IPC资源、cache状态、A0—A4 replacement
  成熟度、依赖边和exclusive critical-path segment；
- 新增physical feature activation ledger，区分IR/Plan/Schedule对象存在与实际驱动dispatch；
- 新增GPU interval union、critical-path closure、`<=3%` residual门禁、joint Amdahl和累计/
  leave-one-out interaction聚合；
- 新增contract-only generate/replay runner，绑定raw、summary、code revision和文件digest；同步更新摘要
  与manifest digest仍会被raw语义重算拒绝；
- 新增20项定向测试；production executor、TIR、runtime默认值均未修改。

## Validation

- 文档作用域只读审计发现12个当前指令风险点，修订清单已纳入本轮；
- 代码盘点确认当前G1仅hook `_run_selected_crown`，native queue主体仍为eager PyTorch，shared
  Task/Schedule在执行后lower，full-stack执行尚未激活；
- RVIR盘点确认replacement executor不存在，必须先完成RVIR-v3 executable payload/mutation contract；
- targeted=`20 passed in 1.07s`；
- 激活 `env.sh` 后全量=`1079 passed, 3 skipped in 372.54s`；首次未加载activation hook的尝试在
  collection阶段因`ModuleNotFoundError: tvm`停止，未产生代码失败，随后按仓库环境合同重跑通过；
- Black check通过；合同、runner与测试三个文件mypy（`--follow-imports=skip`）clean；Pylint=`10.00/10`；
  `git diff --check`通过；
- FSG0状态=`VALIDATED`，仍为`performance_claimed=false`。

## External Audit Response

外部独立审计结论为`APPROVE-WITH-MINOR`（0 blocker / 0 major / 3 minor）。三项均已关闭：

1. PLAN四轴枚举已逐值对齐代码规范，明确功能值与`setup/unclassified`等哨兵值；
2. 测试中的聚合对象用显式`cast`收窄，mypy从2个源文件扩大到合同、runner、测试3个文件；
3. replay实时校验`git_head`，新增同步更新manifest hash后的伪造HEAD拒绝测试。

审计原文归档于`external_audit_fsg0_full_stack_gpu_baseline_2026_08_06.md`；原文不回写，以上为
executor后续响应。

## Decisions

- `1.0764x`只标记selected-CROWN deletion-only ceiling，不是BoundFlow full-stack ceiling；
- 单region share仅用于工程优先级或关闭该专属实现，不再作为整条系统路线kill gate；
- 最终`1.20x queue/1.15x complete-query`只施加到累计B7 vs B0；
- diagnostic native-vs-official不同算法数据永不升级为same-solver speedup；
- 历史artifact的`gpu-winner-reselection`字段不改，文档标记为已被本路线取代。

## Follow-Ups

1. 接official control observer，生成五fresh B0 full-stack baseline；
2. 设计并实现RVIR-v3 executable payload与BoundFlow replacement correctness；
3. correctness关闭后才运行B0/B1/B2 paired timing；
4. 逐层实现B3—B7并做累计与leave-one-out消融。

## Links

- plan: [Full-stack GPU plan](BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md)
- roadmap: [GPU compiler acceleration research](BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md)
