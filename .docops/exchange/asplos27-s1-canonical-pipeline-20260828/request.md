# ASPLOS27 S1 canonical CIBC implementation and formal audit

- task: asplos27-s1-canonical-pipeline-20260828
- doc: asplos27-s1-canonical-pipeline-20260828/request
- from: codex -> to: external-model
- executor: codex / auditor: external-model
- base commit: 1822390
- created: 2026-08-28T01:03:40Z

## Original request

---
status: ready-for-external-audit
date: 2026-08-28
type: external-audit-handoff
topic: boundflow
slug: asplos27-s1-combined-implementation-formal
executor: codex
external-audit: requested
performance-claimed: false
---

# ASPLOS’27 S1 canonical CIBC pipeline：实现与formal合并外审交接

## 1. 请求结论

请外部审计方不要采信本文件数字，独立检查source、raw、hash、语义、结构、性能派生、replay、tamper、
测试和claim边界，并给出：

```text
approve | approve-with-minor | reject
blocker / major / minor / info
AC1—AC8逐项PASS/FAIL及独立证据
```

若AC1—AC8均通过，建议关闭：

```text
VALIDATED-S1-CIBC-CANONICAL-PIPELINE
```

只开放S2 coarse CROWN/custom VJP canonical region。不得据此开放same-solver、complete-query、总体10×或
paper performance claim。

## 2. 审计范围与先决条件

- branch=`feat/rvir-v4-production-state-ownership-v1`；
- S0 closure source=`1822390`，既有外审报告=
  `gemini_doc/external_audit_asplos27_s0_2026_08_27.md`；
- S1 implementation=`aa537ed`；
- S1 artifact-test anchor=`56c494f`；
- 审计时以已推送branch HEAD为文档/DocOps closure点，必须是`56c494f`的后继；
- formal artifact=`artifacts/asplos27-s1-cibc-pipeline/resnet2b-prop0-v2`；
- `v1`是source identity修正前的superseded诊断artifact，不用于结论；
- model/input frozen SHA256=
  `791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d` /
  `f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc`。

不在本轮范围：GC0-1历史exchange、S2代码、αβ-CROWN/BaB替换、held-out泛化、内存收益、总体10×。

## 3. 执行方声明（必须独立重算）

### 3.1 Canonical path

```text
BFTaskModule interval source
  → batch-6 storage specialization
  → one standard Relax 17-op dataflow function
  → 6 paired-output CIBC Conv call_tir
  → 2 cuBLAS codegen partitions covering Linear shape families
  → compiled VM executable
  → prepare-only DLPack/static parameters/views
  → one static-address CUDA Graph invocation
  → graph-stable final lower/upper views
```

正式warm path不得执行旧Plan/Task/Schedule Python interpreter、per-op DLPack construction、fallback、eager
shadow或output materialization copy。

### 3.2 Formal headline

六个fresh process覆盖`BDP/BPD/DBP/DPB/PBD/PDB`六全排列，每进程30 groups、每对象200 replay；
B=PyTorch CUDA Graph、D=direct CIBC CUDA Graph、P=canonical pipeline CUDA Graph，三侧均计入input copy。

- D/B geomean=`2.502345972580691x`，worst=`2.4576329118907867x`；
- P/B geomean=`2.502809985423115x`，worst=`2.460020550139487x`；
- D/P propagation geomean=`1.0001854311304303x`，worst=`0.9898443431059027x`；
- final lower/upper max diff=`0.000244140625`，allclose/sign exact；
- gate：P/B geomean `>=2.20x`、worst `>=2.00x`、propagation geomean `>=0.90x`；
- 17 ops、6/6 CIBC、2 cuBLAS、fallback/eager/warm-DLPack=`0`；
- `s1_performance_admitted=true`，但`performance_claimed=false`。

### 3.3 Frozen hashes

- source=`56c494f1391d49fe24db7bbc9dfab3f7642b5749`；
- protocol=`a6d04d779149224c23c4b16b64f3a4b23a2542582885b80a45e64a7eefa7bcb2`；
- summary=`7c2fe8b0191514bbf70c70528ce459594e8e7846484f596357ffbfe64040ff60`；
- manifest=`bd4eaa4a9f0610d2db9fb8848e27de41ef906372fb96bc03c8317a31260680cc`；
- tamper=`8/8 rejected`，每项均在修改后重新签worker外层hash。

## 4. 审计AC

### AC1：Git顺序与范围

- `aa537ed`只实现S1 canonical pipeline、runner、tests、TVM cuBLAS环境门禁及预注册文档；
- `56c494f`只增加formal artifact replay/tamper tests；
- 后继closure修复只允许historical Git-object replay修正、文档、DocOps和测试；
- production代码不硬编码model/property路径；第三方submodule无源码改动；
- S0已在实现前关闭，S2没有越序实现。

### AC2：不是runner套壳

亲读lowering/runtime，确认P侧真实构造一个Relax function，6个Conv真实`call_tir`，Linear真实进入
cuBLAS codegen，VM/CUDA Graph真实执行该compiled artifact。确认旧direct CIBC对象只用于D oracle，P侧
没有调用它或PyTorch graph作为shadow/fallback。

### AC3：语义与结构

- 独立解析worker六组final lower/upper metrics；
- 检查source task 17-op、residual/fanout拓扑、6 Conv inventory、2 cuBLAS receipt；
- 检查paired-output TIR的center/deviation语义与lower/upper构造；
- 检查schedule缺失、未知op、非法threads、batch不匹配、input mutation均在launch前fail closed；
- 注意：17个中间pair逐层比较只属于实现诊断，**未冻结为formal raw**；不得把它升级为独立artifact证据。

### AC4：Prepared runtime与identity

- parameter pointer/version、input admission pointer/version、source/storage/plan/Relax/lowered/device hashes；
- DLPack只在prepare创建，warm count=0；
- CUDA Graph output为graph-stable view，无额外copy；
- receipt必须拒绝`performance_claimed=true`、cuBLAS=0、coverage/fallback/eager等篡改；
- `historical_sha256(source,path)`必须始终读Git object，即使source等于当前HEAD且worktree脏。

### AC5：独立性能重算

只用Python stdlib解析6 raw：

1. 逐worker从30 groups重算B/D/P median；
2. 重算D/B、P/B、D/P；
3. 重算6-run geomean与worst；
4. 检查六全排列、30×200、三侧CUDA Graph与input-copy字段；
5. 检查门槛是预先冻结且未因结果修改。

不得用summary中的派生数字作为输入。

### AC6：Replay与tamper

- 重算protocol、每worker、summary、manifest全部hash；
- 运行官方replay，必须退出0且输出hash逐位一致；
- 亲读replay确认从raw重算而非只比文件digest；
- 独立新增至少2个fully re-signed变体，建议修改一个group latency和一个receipt/module identity；
- 既有8类：semantic、sign、fallback、cuBLAS、Conv coverage、DLPack、claim、order，必须全拒绝。

### AC7：测试与工具链

执行方结果：targeted=`12 passed`；full=`1868 passed, 3 skipped`。请现场重跑并核对skip原因。
同时检查black、mypy、pylint、TVM CUDA/cuBLAS smoke和`git diff --check`。

### AC8：Claim discipline

权威README/plan/claims map/memo/current status必须一致：

- 可以说S1 standalone IBP compiler-plumbing qualification通过；
- 不可以说αβ-CROWN、BaB、same-solver、query/queue、held-out、memory、总体10×或ASPLOS-ready；
- MR1 production activation exact-call `0/51 eligible`没有被覆盖；
- `performance_claimed=false`全链路保持；
- 下一只允许S2 coarse CROWN/custom VJP，不允许直接跳same-solver formal。

## 5. 推荐命令

```bash
source /home/lee/miniconda3/etc/profile.d/conda.sh
conda activate boundflow
source env.sh

git status --short
git log --oneline --decorate -8
git diff 1822390..HEAD --stat
git show --stat aa537ed
git show --stat 56c494f

python scripts/run_asplos27_s1_cibc_artifact.py \
  --artifact artifacts/asplos27-s1-cibc-pipeline/resnet2b-prop0-v2 --replay

pytest -q tests/test_asplos27_s1_cibc_pipeline.py \
  tests/test_cibc_ibp_conv.py tests/test_cibc_ibp_graph.py
pytest -q tests

python -m black --check --fast \
  boundflow/backends/tvm/relax_interval_task_ops.py \
  boundflow/backends/tvm/cibc_ibp_conv.py \
  boundflow/runtime/asplos27_s1_cibc_pipeline.py \
  scripts/run_asplos27_s1_cibc_worker.py \
  scripts/run_asplos27_s1_cibc_artifact.py \
  tests/test_asplos27_s1_cibc_pipeline.py
python -m mypy \
  boundflow/backends/tvm/relax_interval_task_ops.py \
  boundflow/backends/tvm/cibc_ibp_conv.py \
  boundflow/runtime/asplos27_s1_cibc_pipeline.py \
  scripts/run_asplos27_s1_cibc_worker.py \
  scripts/run_asplos27_s1_cibc_artifact.py
python -m pylint \
  boundflow/backends/tvm/relax_interval_task_ops.py \
  boundflow/backends/tvm/cibc_ibp_conv.py \
  boundflow/runtime/asplos27_s1_cibc_pipeline.py \
  scripts/run_asplos27_s1_cibc_worker.py \
  scripts/run_asplos27_s1_cibc_artifact.py \
  tests/test_asplos27_s1_cibc_pipeline.py
```

## 6. 已知限制（不是finding自动豁免）

1. 只有一个ResNet2B/property、standalone IBP graph；
2. formal需要RTX 4060 Laptop GPU、TVM CUDA与`USE_CUBLAS=ON`；
3. compile约`0.75–0.79 s`，未计入warm headline，已单独披露；
4. CUDA Graph要求静态shape/pointer/schedule；
5. 17个中间pair诊断未冻结raw，formal correctness只对final pair可独立重放；
6. artifact目录通常不进Git，审计机器必须能访问executor侧冻结artifact；
7. 本轮没有same-solver、complete-query或总体10×测量。

## 7. 审计产出格式

请将完整报告写入：

```text
gemini_doc/external_audit_asplos27_s1_combined_2026_08_28.md
```

报告必须分开列出“独立复算”“源码检查”“现场重跑”“仅由冻结raw可核”“不可现场核验”，并明确是否
同意关闭S1、是否只开放S2，以及所有finding的严重级别。


## Scope

S1 canonical Relax/TIR/prepared runtime implementation, v2 formal artifact, replay/tamper, tests, docs and claim boundary

## Acceptance criteria

- AC1 git order and scope
- AC2 canonical compiled path is not a runner wrapper
- AC3 semantic and structural correctness
- AC4 prepared runtime and immutable identity
- AC5 stdlib-only raw performance recomputation
- AC6 replay and fully re-signed tamper rejection
- AC7 targeted/full/static/toolchain validation
- AC8 claim discipline and S2-only successor
