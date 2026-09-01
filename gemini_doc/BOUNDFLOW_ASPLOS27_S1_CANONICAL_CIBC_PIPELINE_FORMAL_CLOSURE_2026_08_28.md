---
status: validated-s1-cibc-canonical-pipeline
date: 2026-08-28
type: changelog
topic: boundflow
slug: asplos27-s1-cibc-formal-closure
external-audit: pending-combined-review
performance-claimed: false
---

# ASPLOS’27 S1 canonical CIBC pipeline formal closure

## Verdict

`VALIDATED-S1-CIBC-CANONICAL-PIPELINE`。

完整17-op IBP图已通过一个standard Relax function和一个prepared VM/CUDA-Graph入口执行，并在六个fresh
process中基本无损保住旧direct-CIBC winner。该结论只开放S2实现，不是same-solver、BaB、complete-query或
ASPLOS headline claim。

## Frozen identity

- source=`56c494f`（实现=`aa537ed`，artifact replay tests=`56c494f`）；
- artifact=`artifacts/asplos27-s1-cibc-pipeline/resnet2b-prop0-v2`；
- protocol hash=`a6d04d779149224c23c4b16b64f3a4b23a2542582885b80a45e64a7eefa7bcb2`；
- summary hash=`7c2fe8b0191514bbf70c70528ce459594e8e7846484f596357ffbfe64040ff60`；
- manifest hash=`bd4eaa4a9f0610d2db9fb8848e27de41ef906372fb96bc03c8317a31260680cc`；
- source/model SHA256=`f42229dd…126dc`/`791aa24d…4a6d`；
- hardware=`NVIDIA GeForce RTX 4060 Laptop GPU / sm_89`；
- environment=`Python 3.12.12 / torch 2.12.1+cu132 / TVM 0.23.dev0 / CUDA 13.2`。

## Canonical execution path

```text
ONNX/Primal interval task
  → exact batch-6 specialization
  → Relax 17-op dataflow
  → CIBC paired-output TIR: 6 Conv
  → cuBLAS codegen partitions: 2 Linear shape families / 8 calls
  → VM executable
  → prepare-only DLPack views and versioned admission
  → static-address CUDA Graph
  → graph-stable final lower/upper views
```

旧Plan/Task/Schedule不在warm path解释执行；Task只作为import compatibility source。production replay内
per-op Python、DLPack view construction、fallback、eager shadow和output materialization copy均为0。

## Six-fresh result

每个worker按指定顺序对B=PyTorch、D=direct-CIBC、P=canonical pipeline各执行30组×200 replay；三侧都
是CUDA Graph并计入lower/upper input copy。

| run | order | B ms | D ms | P ms | D/B speedup | P/B speedup | P/D propagation |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | BDP | 0.172357 | 0.070116 | 0.070048 | 2.458150x | 2.460554x | 1.000978x |
| 1 | BPD | 0.173079 | 0.070385 | 0.070302 | 2.459039x | 2.461931x | 1.001176x |
| 2 | DBP | 0.180388 | 0.070971 | 0.071253 | 2.541719x | 2.531668x | 0.996046x |
| 3 | DPB | 0.173148 | 0.070453 | 0.070385 | 2.457633x | 2.460021x | 1.000972x |
| 4 | PBD | 0.184418 | 0.072975 | 0.072093 | 2.527142x | 2.558056x | 1.012233x |
| 5 | PDB | 0.186701 | 0.072562 | 0.073307 | 2.572979x | 2.546849x | 0.989844x |

派生结果：

- direct/PyTorch geomean=`2.5023459726x`，worst=`2.4576329119x`；
- pipeline/PyTorch geomean=`2.5028099854x`，worst=`2.4600205501x`；
- pipeline/direct geomean=`1.0001854311x`，worst=`0.9898443431x`；
- 三条门槛`2.20x / 2.00x / 0.90x`全部通过。

## Semantics and failure boundaries

- 三方final lower/upper max diff=`0.000244140625 <=3e-4`，allclose/sign exact；
- 实现诊断时用debug Relax module逐层导出17个中间pair，residual/fanout均闭合，最大差在final Linear
  出现；该逐层诊断**未冻结为formal raw**，formal可重放证据只覆盖final pair、17-op/6-Conv结构计数与
  source/lowered IR identity，外审不得把前者误记为独立artifact结论；
- compile receipt绑定source task、parameter contents、specialized storage、plan、source/lowered Relax、device
  source与target；
- dynamic input先做finite/lower≤upper admission，warm仅检查pointer/version；mutation在launch前拒绝；
- 6/6 CIBC call_tir、2 cuBLAS partitions、fallback/eager=0、warm DLPack=0；
- root replay从raw重算median、三组speedup/geomean/worst与门禁，退出0；
- 8/8 fully outer-resigned tamper rejected：semantic、sign、fallback、cuBLAS、Conv coverage、DLPack、claim、order。

## Claim boundary and next action

允许声明：在单GPU、单模型/property、standalone IBP图上，canonical Relax/TIR/prepared pipeline保住了已有
CIBC winner，compiler plumbing qualification通过。

禁止声明：αβ-CROWN/BaB/query/complete-query speedup、跨模型泛化、memory收益、10×总体结果或
ASPLOS-ready。MR1的production activation exact-call `0/51` eligible仍成立。

唯一下一动作：S2 coarse CROWN/custom VJP region复用同一Relax/TIR/prepared runtime；S2性能仍需同scope
direct/native/cumulative实测，不能把本阶段`2.5028x`代入query。

## Validation ledger

- targeted S1/CIBC tests：`12 passed`；
- full suite：`1868 passed, 3 skipped`；skip为1个“TVM已存在时跳过no-TVM重复编译”与2个冻结
  VNN-COMP checkout缺失，均非S1回归；
- replay：v2 raw→summary→manifest退出0；
- tamper：8/8 fully outer-resigned变体被拒；
- 历史identity读取专项：校验器即使source等于当前HEAD，也强制读取commit object而非dirty worktree；
- `performance_claimed=false`保持。
