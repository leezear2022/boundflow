---
status: ready-for-external-audit
updated: 2026-08-23T07:05:00Z
type: audit-handoff
topic: boundflow
slug: fsg4-b4b2-b2-2-sparse-source-external-audit
stage: s01
---

# FSG4/B4-B2 B2-2 Sparse-source Linear TIR External Audit Handoff

## 1. 冻结点与状态上限

- branch=`feat/rvir-v4-production-state-ownership-v1`；
- source=`8bd1db2`；base=`a6ca7f8`（B2-1 外审关闭）；
- B2-1 implementation=`eb74e45`；B2-1 handoff=`2da99da`；
- preregistration=`57be636`；
- 状态上限=
  `VALIDATED-B4-B2-B2-2-SPARSE-SOURCE-CORRECTNESS-PENDING-EXTERNAL-AUDIT`；
- 本轮无 timing、P-anchor、B2-3/B2-4/B2-5、same-solver 或 B4-B3。

审计必须以`8bd1db2`为 clean code source。后续纯外审交接提交不能替换 source；任何产品代码变化
都必须重新声明 source 并重跑全部门禁。

## 2. 变更范围

新增代码：

- `boundflow/ir/differentiable_lower_sparse_linear_tir.py`；
- `boundflow/backends/tvm/differentiable_lower_sparse_linear.py`；
- `boundflow/runtime/fsg4_b4b2_sparse_linear_tir.py`；
- `scripts/run_fsg4_b4b2_sparse_linear_tir_correctness.py`；
- `tests/test_fsg4_b4b2_sparse_linear_tir.py`。

其余均为状态/变更文档。`git diff a6ca7f8 8bd1db2`不得修改 B2-0/B2-1 product path、vendored
TVM、production exact-call、optimizer、Conv/P-anchor 或 timing infrastructure。

## 3. 不可信执行方摘要

以下只作为待独立复核声明，不得直接采信：

- raw/run/metric/element=`5/5/20/31,590`；
- max abs diff=`8.642673492431641e-07`，allclose/sign exact=`true/true`；
- compressed alpha=`[6,27]`，compressed beta=`[6,1]`；
- template=`adddcb6a5daa7ebf8a8dcc34cc0e08b1f2a30dd6ad43503f2ab7f3df2b9bf56f`；
- schedule=`b8fe0a7d2f859ada4f1bf3293b80ba6783003861ed66a16ad0a5542cc2350d57`；
- module receipt=`7f6ab5cbfceaaa8b29529d0624e9238f1f52386ad1279ee24d52b31d6f842679`；
- workspace=`adjoint_matmul,output_bias_delta`，forbidden count=`0`；
- DLPack=`21/21`，cache=`miss,hit,hit,hit,hit`，launch=`1/1`，fallback/eager=`0/0`；
- targeted=`34 passed`，B4-B related=`88 passed`，full=
  `1448 passed, 3 skipped, 6 warnings`。

## 4. Acceptance criteria

### AC1：顺序、source、scope

独立确认：

1. B2-1 外审 approve/关闭早于 B2-2 实现；
2. source/base/prereg commits 存在，HEAD/origin 与交接声明一致；
3. `a6ca7f8..8bd1db2`只有 5 个新增代码文件与状态文档；
4. 无 P-anchor/Conv、timing/performance、B2-3+、production path、optimizer 或 vendored 改动；
5. `.docops/s.md`只开放 B2-2 外审。

### AC2：compressed mapping 与 first-class identity

不要信任 Template 中的 tuple/hash；直接读取 5 个
`artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1/run_0*.pt` 的 S capture，独立确认：

1. production alpha=`[2,1,6,27]`，本 ABI 只消费 direction/spec=`[0,0]`后的`[6,27]`；
2. alpha feature index 恰 27 个、严格递增、唯一、范围`[0,100)`；
3. production beta=`[6,1]`，location=`[6,1]`、sign=`[6,1]`且 sign 只为`{-1,1}`；
4. mapping 常量进入 Template canonical JSON/stable hash/cache key，动态 tensor 不进入 compile key；
5. 独立重算 Template/Schedule/Module 三项 hash，与§3逐位一致；
6. Template/Instance/Schedule/Module/Projection/Launch round-trip 全部成立；
7. 合法重签但改变 mapping、ABI、gradient inventory、workspace、toolchain、symbol、counter 或 claim
   的对象均 fail closed。

### AC3：独立 sparse-source 数学复核

不得把 B2-1 dense TIR 当唯一 oracle，也不得只调用执行方 metric helper。审计方至少以独立 PyTorch
或 float64 表达式，从 compressed alpha/beta 与 mapping constants 直接重写：

1. 未映射 alpha feature 的 native lower slope source 必须为 0；
2. 映射 alpha feature 直接读取 compressed value 并保持 clamp 端点导数所有权；
3. beta pre-add 只在每 domain 冻结 location 上为`-compressed_beta*sign`；
4. Linear forward/output-bias 直接对 B4-B1 pure-PyTorch oracle；
5. compressed alpha gradient 等于 native oracle 在 27 coordinates 的 gather；
6. compressed beta gradient 等于 native oracle 在 6 locations 的 gather；
7. scatter-back candidate 与 native oracle 在 owned coordinates 一致，unowned native gradient 为 0；
8. 每 raw 的 4 metrics、总 31,590 elements 全部 finite、`atol=rtol=2e-4`、nonzero sign exact；
9. 单独核对 alpha=0/1 与 A==0 的离散导数，以及 beta gradient 6/6 nonzero。

### AC4：scheduled TIR 与 workspace 真实性

亲读并现场构建 unscheduled/scheduled TIR：

1. forward/backward 输入中不存在 native dense alpha/beta；
2. backward 输出直接为`[6,27]`/`[6,1]`，不存在 native dense gradient 后置 gather；
3. mapping 是 Template 编译常量，不是 timed-region 外的 runtime dense scatter；
4. scheduled TIR 中只观察到`adjoint_matmul`与`output_bias_delta` workspace；
5. `native_alpha`、`native_beta`、`scaled_a`、`relu_lower_a`全局 workspace occurrence=`0`；
6. Module receipt 的 observed/forbidden ledger 必须由实际 scheduled TIR 重算，而非只信 schedule 声明。

### AC5：runtime、projection 与失败路径

现场 GPU 复跑并确认：

1. forward/backward 各恰一次 PackedFunc launch；
2. 21/21 DLPack data_ptr exact，4 outputs 不 alias 10 inputs；
3. current Torch stream 与 TVM-FFI raw stream 双向一致，非默认 stream 可通过；
4. cold miss 后 4 次 warm hit，动态值不改变 module identity；
5. fallback/eager 是真实计数器，reject 先计数再 raise；
6. dtype/device/nonfinite/range、mapping、adjoint pointer、重复 launch、higher-order 全部 fail closed；
7. missing-symbol 异常后 device/current stream/deterministic enabled/debug mode 原样恢复；
8. Projection receipt 同时绑定 native/compressed/candidate/scatter hashes，mapping exact 与数值 tolerance
   pass 不得混为 bitwise equality。

### AC6：runner、回归与静态

至少现场运行：

```bash
source /home/lee/miniconda3/etc/profile.d/conda.sh
conda activate boundflow
python scripts/run_fsg4_b4b2_sparse_linear_tir_correctness.py
python -m pytest -q \
  tests/test_fsg4_b4b2_identity_tir.py \
  tests/test_fsg4_b4b2_dense_linear_tir.py \
  tests/test_fsg4_b4b2_sparse_linear_tir.py
python -m pytest -q tests/test_fsg4_b4b*.py
python -m pytest -q -rs
python -m black --check \
  boundflow/ir/differentiable_lower_sparse_linear_tir.py \
  boundflow/backends/tvm/differentiable_lower_sparse_linear.py \
  boundflow/runtime/fsg4_b4b2_sparse_linear_tir.py \
  scripts/run_fsg4_b4b2_sparse_linear_tir_correctness.py \
  tests/test_fsg4_b4b2_sparse_linear_tir.py
python -m mypy \
  boundflow/ir/differentiable_lower_sparse_linear_tir.py \
  boundflow/backends/tvm/differentiable_lower_sparse_linear.py \
  boundflow/runtime/fsg4_b4b2_sparse_linear_tir.py \
  scripts/run_fsg4_b4b2_sparse_linear_tir_correctness.py
python -m pylint \
  boundflow/ir/differentiable_lower_sparse_linear_tir.py \
  boundflow/backends/tvm/differentiable_lower_sparse_linear.py \
  boundflow/runtime/fsg4_b4b2_sparse_linear_tir.py \
  scripts/run_fsg4_b4b2_sparse_linear_tir_correctness.py \
  tests/test_fsg4_b4b2_sparse_linear_tir.py
bash scripts/rebuild_tvm.sh
python /home/lee/.codex/plugins/cache/personal/docops-logic/0.2.0+codex.20260802165548/scripts/dol.py lint --soft
```

3 个 skip 必须逐项核对为既有环境边界，不能只报告总数。

### AC7：claim 边界与下一动作

确认新代码无 timing API，`performance_claimed=false`，并且：

- `sparse_source_admitted=true`只证明 compressed-source ABI correctness；
- 不得外推 speedup、memory ratio、P-anchor、whole-core/query、B0 parity、same-solver 或 ASPLOS-ready；
- 只有 approve 且 0 blocker/major 才允许关闭 B2-2；
- 审计结果最多开放 B2-3 P-anchor dense correctness，不能开放 timing、B2-4/B2-5 或 B4-B3。

## 5. 已知边界

- 5 个 S captures 的数值确定性重复，不是 5-process performance 证据；本轮不计时；
- stdout 未冻结 artifact，formal raw/replay/tamper 仍属于 B2-5；
- `adjoint_matmul`与`output_bias_delta`仍是 dense workspace，本轮 claim 只排除 native alpha/beta/
  scaled-A/relu-A materialization；
- 本轮只覆盖 S Linear，不覆盖 P Conv、incoming-A gradient 或物理门禁；
- execution/projection receipts 是 correctness evidence，不是 production activation receipt。

## 6. 审计输出格式

报告请放在`gemini_doc/`，至少包含：

1. verdict：`approve`或`request_changes`；
2. blocker/major/minor/info findings；
3. AC1—AC7逐条 PASS/FAIL 与独立证据；
4. 独立 sparse-source 重算结果、mapping 与 workspace 复核；
5. 现场 GPU、targeted/related/full/static/DocOps 结果；
6. claim 边界与下一动作判定。

只有`approve`且无 blocker/major，才允许关闭 B2-2 并开放 B2-3 P-anchor dense correctness。
timing、B2-4/B2-5 与 B4-B3 在任何情况下仍关闭。

