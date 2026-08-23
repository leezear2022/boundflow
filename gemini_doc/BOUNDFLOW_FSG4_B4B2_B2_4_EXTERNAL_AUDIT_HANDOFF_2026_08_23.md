---
status: ready-for-external-audit
updated: 2026-08-23T13:57:44Z
type: handoff
topic: boundflow
slug: fsg4-b4b2-b2-4-external-audit-handoff
stage: s01
---

# FSG4/B4-B2 B2-4 最终外部审计交接

## 1. Frozen Audit Target

- branch=`feat/rvir-v4-production-state-ownership-v1`；
- base=`b18fad483fcfa9bbef61337628f368a7ca2fd7c2`（B2-3外审正式关闭）；
- source/result commit=`1f8d47a8acd55f9b315e207a549f515e29a6f35e`；
- internal status=
  `VALIDATED-B4-B2-B2-4-SPARSE-CONV-P0-AND-BOUNDED-LEDGER-CORRECTNESS-PENDING-EXTERNAL-AUDIT`。

这是B2-4唯一且最终的外审，不拆P0/ledger中间轮次。请不要采信summary数字，应从raw、mapping、
scheduled TIR、GPU执行与git事实独立复核。

## 2. Scope and Non-scope

本轮范围：

- P-anchor compressed-alpha/empty-beta sparse-source Conv forward/backward；
- P0 five-raw correctness；
- 12项预登记schedule candidate ledger、每项compile/correctness与hash唯一性；
- receipt、workspace、gradient projection、ABI与负向门禁。

本轮明确不含：

- 任何CUDA event或CPU wall timing；
- winner selection、speedup、memory saving；
- B2-5 formal independent-process artifact/replay；
- B4-B3、whole-core/query、B0 parity或ASPLOS-ready。

## 3. Changed Files

核心新增：

- `boundflow/ir/differentiable_lower_sparse_conv_tir.py`；
- `boundflow/backends/tvm/differentiable_lower_sparse_conv.py`；
- `boundflow/runtime/fsg4_b4b2_sparse_conv_tir.py`；
- `scripts/run_fsg4_b4b2_sparse_conv_tir_correctness.py`；
- `tests/test_fsg4_b4b2_sparse_conv_tir.py`。

附带：

- `tests/test_fsg4_b4b2_dense_conv_tir.py`补独立shape-mismatch拒绝测试；
- 计划/claims/status/change record/DocOps更新。

无production、optimizer、vendored TVM/TVM-FFI/auto_LiRPA改动。

## 4. AC1 — Git Order, Scope and Preregistration

独立核对：

1. `1f8d47a`后继于已approved/closed的B2-3 commit `b18fad4`；
2. diff只实现B2-4与文档，没有timing/B2-5/B4-B3越序；
3. 预注册knob集合与ledger≤12门禁未在结果后修改；
4. 新增结果段不能覆盖历史门禁；
5. `performance_admitted/claimed=false`、`timing_raw_present=false`、
   `winner_selected=false`均fail closed。

## 5. AC2 — Production Sparse Mapping and Empty Beta

从5份`run_XX.pt`的P capture独立解析：

- production alpha=`[2,1,6,86]`；direction/spec选择=`[0,0]`；
- 三个feature-index tensor各`[86]`，形成86个唯一合法`(c,h,w)`；
- compressed alpha ABI=`[6,86]`；
- production beta/location/sign均为`[6,0]`；
- `beta_active=false`、reference beta gradient=`None`；
- reference native alpha gradient在516个owned元素外严格为零。

检查runtime/TIR输入中不存在native dense alpha、compressed beta、native beta或scatter workspace；
mapping坐标必须是template常量并进入stable/cache hash。

## 6. AC3 — Independent Mathematical Oracle

不要把dense B2-3或repo B4-B1 reference作为唯一oracle。建议以float64、无autograd闭合式独立重算：

- compressed coordinate lookup/scatter semantics；
- lower/upper slope与selected intercept；
- ConvTranspose forward与adjoint索引；
- output bias reduction；
- compressed alpha VJP（直接返回`[6,86]`）；
- incoming-A VJP；
- unowned native gradient zero与beta gradient absent。

对5 raw P0四路输出逐元素核对：`output_lower_a`、`output_bias`、
`compressed_alpha_gradient`、`incoming_lower_a_gradient`。门禁=`atol/rtol 2e-4`、finite、sign exact。

## 7. AC4 — Live P0 Five-raw GPU Gate

现场执行：

```bash
source /home/lee/miniconda3/etc/profile.d/conda.sh
conda activate boundflow
python scripts/run_fsg4_b4b2_sparse_conv_tir_correctness.py
```

P0应独立复现：

- run/metrics/elements=`5/20/64,050`；
- max diff=`2.384185791015625e-06`，allclose/sign exact；
- cache=`miss,hit,hit,hit,hit`；
- DLPack=`19/19`，launch=`1/1`，fallback/eager=`0/0`；
- beta gradient absent；projection owned=`516`、mapping exact、unowned zero；
- template=`c51b77cbdf28551cb8b97252d82a5abdda76851c5fb49e0a54547ac898f14075`；
- P0 schedule=`a4937031317c43e3e87567eacf341029813cc4bad52a4aa15708feff83d855f1`；
- P0 module=`44800f32e23693881cc7515cc8a4048eb005e1893e9a1d97fa4a7cc30851fce4`。

## 8. AC5 — Twelve-candidate Ledger

独立验证ledger恰为12项，ordinal连续且knobs为：

```text
0  (128,16,1,1)   1  (256,16,1,1)
2  (128, 8,1,1)   3  (256, 8,1,1)
4  (128, 4,1,1)   5  (256, 4,1,1)
6  (128,16,2,1)   7  (256,16,2,1)
8  (128, 8,2,1)   9  (256, 8,2,1)
10 (128,16,1,3)  11  (256,16,1,3)
```

要求：

- 12个schedule hash唯一，ledger hash=
  `1660edca9f23201b14edfe8ce06947ec16f52b5b311ddb47174ea1955e8d07c6`；
- 12个scheduled TIR结构确实不同，不只是receipt字段不同；
- 12个module receipt hash唯一；
- 每项独立compile并执行capture0四路correctness，12/48 metrics/153,720元素全过；
- 全候选max diff=`2.384185791015625e-06`；
- candidate cache event均为`miss`，不得复用P0 module冒充；
- ledger在任何timing raw前冻结，不能追加第13项；
- 不得依据测试墙钟、编译墙钟或未登记测量选择winner。

请篡改candidate count/order/knobs/hash、`timing_raw_present`、`winner_selected`与
`performance_claimed`，确认重签外层字段后仍fail closed。

## 9. AC6 — Structural Workspace and Physical Schedule

直接遍历scheduled TIR `Block.alloc_buffers`，每个候选必须恰为：

```text
adjoint_conv      [6,1,16,8,8]
output_bias_delta [6,1]
```

确认：

- `relu_lower_a`与`adjoint_relu`已inline；
- 无native alpha/beta、compressed beta、scaled-A或scatter global workspace；
- thread extent/channel tile/spatial tile/reduction unroll真实反映在scheduled TIR；
- schedule hash、module hash与cache key一一绑定。

## 10. AC7 — Receipts, Negative Paths and Findings

检查Template/Instance/Schedule/Module/Projection/Launch/Ledger round-trip与篡改拒绝，至少覆盖：

- duplicate/out-of-range coordinates；
- shape/dtype/device/nonfinite/alpha range；
- S-anchor/active-beta/attrs/scope broadening；
- fallback/eager、higher-order、stream、pointer、launch、workspace；
- projection owned count/mapping/numerical/sign/unowned-zero/beta-absent；
- schedule knob与ledger timing/winner/performance claim。

B2-3 finding处置：

1. dense Conv独立shape-mismatch测试应已补齐并真实运行；
2. module TIR/device hash独立重编译比对尚未声称关闭，应明确保留给B2-5 replay。

## 11. AC8 — Validation Chain

复跑：

```bash
pytest -q tests/test_fsg4_b4b2*.py
pytest -q tests/test_fsg4_b4b*.py
pytest -q
pytest -q tests/test_env.py
```

内部结果：

- targeted=`51 passed`；
- B4-B related=`105 passed`；
- full=`1465 passed, 3 skipped, 6 warnings`；
- test_env=`3 passed`；
- Black clean、Mypy 4 source clean、Pylint=`10.00/10`、diff check通过；
- TVM rebuild=`ninja: no work to do`；
- DocOps validate/lint应通过。

## 12. Verdict Boundary

报告写入：

`gemini_doc/external_audit_b4b2_b2_4_sparse_conv_tir_2026_08_23.md`

按AC1–AC8列出PASS/FAIL、独立证据、blocker/major/minor/info与不可现场复核项。

只有`APPROVE`且0 blocker/major，才能关闭B2-4并开放B2-5 formal artifact/timing实现；即使批准，
也不能自动宣称任何speedup、winner、B4-B2 GO或开放B4-B3。B2-5必须使用当前12项冻结ledger，
不得追加候选。
