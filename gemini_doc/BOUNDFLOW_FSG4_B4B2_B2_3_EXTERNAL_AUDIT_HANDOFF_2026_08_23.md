---
status: ready-for-external-audit
updated: 2026-08-23T11:33:28Z
type: handoff
topic: boundflow
slug: fsg4-b4b2-b2-3-external-audit-handoff
stage: s01
---

# FSG4/B4-B2 B2-3 外部审计交接

## 1. 审计对象与冻结边界

- branch=`feat/rvir-v4-production-state-ownership-v1`；
- base=`c28c903`；
- source/result commit=`73070706935f2e6610d4e12903e1d9b4f67b0f83`；
- 内部 claim=
  `VALIDATED-B4-B2-B2-3-P-CONV-DENSE-CORRECTNESS-PENDING-EXTERNAL-AUDIT`；
- 仅审计 P-anchor `performance-conv-8-candidate` dense Conv forward/backward correctness；
- timing、B2-4 sparse-source schedule search、B2-5 formal artifact、B4-B3 均不在本轮范围。

请不要采信 changelog/runner summary 数字；应从冻结 raw、源码、scheduled TIR 和独立 oracle 重算。

## 2. 变更文件

核心新增：

- `boundflow/ir/differentiable_lower_dense_conv_tir.py`；
- `boundflow/backends/tvm/differentiable_lower_dense_conv.py`；
- `boundflow/runtime/fsg4_b4b2_dense_conv_tir.py`；
- `scripts/run_fsg4_b4b2_dense_conv_tir_correctness.py`；
- `tests/test_fsg4_b4b2_dense_conv_tir.py`。

附带修复：

- `tests/test_fsg4_b4b2_dense_linear_tir.py`新增 B2-1 dtype/device/nonfinite 拒绝测试；
- 权威计划、claims/status、change log 与 DocOps state 更新为 B2-3 待外审。

production、optimizer、vendored TVM/TVM-FFI/auto_LiRPA 均无改动。

## 3. 冻结 P-anchor 合同

- input/output A=`[6,1,16,8,8]`；native alpha/lower/upper=`[6,16,8,8]`；
- weight=`[16,16,3,3]`，operator bias=`[16]`；
- stride/padding/dilation=`(1,1)`，output padding=`(0,0)`，groups=`1`；
- incoming-A 与 native alpha gradient 必须 present；
- compressed beta=`[6,0]`，beta gradient 必须 absent，runtime/TIR ABI 中不得出现伪零 beta tensor；
- forward/backward 各 exact launch 1 次，fallback/eager 0 次，current stream 与 DLPack zero-copy exact。

## 4. AC1：范围、顺序与 first-class receipts

独立核对：

1. `7307070`是`c28c903`的后继，且实现前 B2-2 外审已批准；
2. 变更无 timing API、B2-4 schedule search、B2-5 artifact 或 B4-B3；
3. Template/Instance/Schedule/Module/Launch 全部 round-trip、stable hash、fail-closed；
4. `performance_admitted=false`、`performance_claimed=false`不能被重签篡改；
5. S-anchor、非 exact Conv attrs/shape、active beta 必须在 admission 前拒绝。

## 5. AC2：独立数学与数值复核

不要复用 B4-B1 reference 实现作为唯一数学 oracle。建议用 float64、无 autograd 的闭合公式独立重算：

- lower/upper slope 与 selected intercept；
- ConvTranspose forward 索引 `weight[ci,co,kh,kw]`；
- output bias 的 intercept 与 operator-bias reduction；
- Conv adjoint；
- native alpha VJP 的 ambiguous/clamp endpoint 所有权；
- incoming-A VJP=`adjoint_relu*selected_slope + output_bias_grad*selected_intercept`。

对 5 份 raw 的四项输出/梯度逐元素核对：`output_lower_a`、`output_bias`、
`native_alpha_gradient`、`incoming_lower_a_gradient`。验收：atol/rtol=`2e-4`、finite、sign exact；
并确认 reference/native beta gradient 为 `None`。

内部数字仅供对照：5 raw/20 metrics/92,190 elements，max diff=
`2.384185791015625e-06`。

## 6. AC3：现场 GPU 与 ABI

在 RTX 4060 Laptop GPU/sm_89 现场重跑：

```bash
source /home/lee/miniconda3/etc/profile.d/conda.sh
conda activate boundflow
python scripts/run_fsg4_b4b2_dense_conv_tir_correctness.py
```

要求 independently reproduce：

- template=`950f20535ab55120e497401c7d17513c5f2118fd65401e4e87d3a081567c4dc2`；
- schedule=`1de607ad7faf39ff1b45ee81b90013e3cc841c69e97fd3aabba0f135893cc7ec`；
- module receipt=`4511fbc51159cea516e568f025636fa9fee0cf97225f032ddf877f8239dbad79`；
- cache=`miss,hit,hit,hit,hit`；每 run launch=`1/1`、fallback/eager=`0/0`；
- DLPack pointer exact=`19/19`，forward/backward current stream exact；
- beta gradient absent，incoming-A/native alpha gradient present。

## 7. AC4：结构化 workspace 门禁

直接遍历 scheduled TIR `Block.alloc_buffers`，不要只搜索 script 字符串。observed inventory 必须恰为：

```text
adjoint_conv      [6,1,16,8,8]
output_bias_delta [6,1]
```

请篡改 schedule/receipt 的 buffer name、shape、inventory 或
`structural_workspace_check`，确认即使重算外层 receipt hash 仍 fail closed。

## 8. AC5：拒绝路径与审计 finding 关闭

确认专项覆盖并真实触发：

- P-anchor dense Conv dtype/device/nonfinite/alpha range/invalid interval；
- S-anchor/scope broadening；
- missing symbol 异常退出后 device/current stream/determinism policy 不漂移；
- fallback/eager 真实计数后拒绝；
- higher-order gradient；
- instance/module/launch/performance claim 篡改；
- B2-1 dense Linear 的 dtype/device/nonfinite 测试确实关闭上轮 info finding。

## 9. AC6：验证链

至少复跑并记录：

```bash
pytest -q tests/test_fsg4_b4b2_identity_tir.py \
  tests/test_fsg4_b4b2_dense_linear_tir.py \
  tests/test_fsg4_b4b2_sparse_linear_tir.py \
  tests/test_fsg4_b4b2_dense_conv_tir.py
pytest -q tests/test_fsg4_b4b*.py
pytest -q
pytest -q tests/test_env.py
```

内部结果分别为`43 passed`、`97 passed`、`1457 passed, 3 skipped`、`3 passed`。另核对
Black、Mypy、Pylint 10.00、diff check、TVM rebuild 与 `dol lint --soft`。

## 10. 审计输出格式与判定

报告请写到：

`gemini_doc/external_audit_b4b2_b2_3_dense_conv_tir_2026_08_23.md`

按 AC1–AC6 给出 PASS/FAIL、独立证据、blocker/major/minor/info findings、不可现场复核项与最终 verdict。

只有 `APPROVE` 且 0 blocker/major，才能关闭 B2-3 并只开放 B2-4 P-anchor sparse-source
correctness/schedule ledger。即使批准，也不能直接开放 timing、B2-5 或 B4-B3。
