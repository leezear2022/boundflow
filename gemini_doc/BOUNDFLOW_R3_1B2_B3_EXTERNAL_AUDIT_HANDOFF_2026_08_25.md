---
status: ready-for-external-audit
updated: 2026-08-25T06:12:00+08:00
type: handoff
topic: boundflow
slug: r3-1b2-b3-external-audit-handoff
stage: s01
---

# R3-1b2 / b3 External Audit Handoff

## 1. 审计结论请求

请独立判定以下两个关闭是否成立：

1. `VALIDATED-R3-1B2-COMPILED-P-ALPHA-VJP`；
2. `VALIDATED-R3-1B3-COMPILED-FIVE-FRESH`，并据此允许
   `r3_1_admitted=true / r3_2a_open=true`。

不要采信 closure/summary headline；请从 code、raw、protocol 与 physical memory receipt 重算。

## 2. 范围和提交序列

分支：`feat/rvir-v4-production-state-ownership-v1`，PR #60。R3-1b1 closure=`5138980`。

- `2fa8624`：独立P-alpha VJP数学归约；
- `8a2575c`：10-symbol compiled VJP/custom Function；
- `6fbc17f`：b2 artifact/replay/tamper协议；
- `12402da`：formal前修正native CUDA autograd末位hash口径；
- `3b60d4a`：b2 artifact与formal closure；
- `4846c97`：b3 five-fresh预注册/worker/replay/tamper；
- `eeeb1bf`：formal前修正fresh worker TVM路径；
- delivery HEAD：b3 artifact、closure和本handoff。

用户未跟踪文件`docs/CIBC_for_DAC.pdf`不在变更范围，不应进入提交。

## 3. AC1 — 实现边界

检查：

- b1 formal绑定的backend/runtime文件未被后继实现修改；
- b2 backend exact 10 symbols，global workspace=0；
- custom forward使用b1 full-lower并compiled-clear arena；
- backward不调用`torch.autograd.grad`、native oracle、`_evaluate_full_region`或CROWN eager evaluator；
- outer autograd调用只负责触发custom Function；
- reverse coefficient只使用2 scratch；forward到backward saved dense A=0；
- A18/A20/A24/Ainput只保留4个byte sign bitmap，共43,008 B；A26在gradient前重算；
- output exact production compressed `[2,1,6,86]`，P beta absent；
- default stream、identity/tensor/metadata异常fail closed。

建议独立推导 `d lower / d alpha25 = A26 * z25` 的符号/ambiguous/clamp端点所有权，并确认
TIR effective-value重放与production relaxation一致。

## 4. AC2 — b2 raw/replay

artifact：`artifacts/r3-structured-owner/r3-1b2-compiled-p-alpha-vjp-v1/`

请独立重算6个lower和1,032个dα元素：

- candidate lower/dα float32 hash=`caa90002...cbb` / `59a35857...813`；
- max diff=`3.814697265625e-06 / 6.146728992462158e-08`；
- sign exact，dα nonzero=`281/281`；
- launches=`15 forward b1 / 15 backward b1 / 10 b2`；
- saved dense A=0，warm dynamic allocated=0，DLPack=`79/79 + 1/1`；
- replay exact，12/12 fully re-signed tamper rejected。

注意：native PyTorch CUDA autograd raw末位跨fresh process可变。协议冻结每份native raw的自洽hash，
再重算tolerance/sign/nonzero；candidate hash仍逐位冻结。请判断该口径是否合理且未弱化语义门禁。

## 5. AC3 — b3 fairness / physical memory

artifact：`artifacts/r3-structured-owner/r3-1b3-compiled-five-fresh-v1/`

请从10个`.pt` raw独立重算：

- pair order=`NC/CN/NC/CN/NC`；
- 每个mode确为fresh subprocess；
- candidate preparation发生在reset前，但其live PlanInstance storage包含在absolute baseline/peak；
- headline使用absolute peak，不是increment；
- 五个native peak allocated/reserved均=`18,487,296 / 25,165,824 B`；
- 五个candidate均=`1,186,304 / 4,194,304 B`；
- worst ratio=`0.06416860529522543 / 0.16666666666666666`；
- all semantic/structure/allocated/reserved=true；
- replay exact，9/9 fully re-signed tamper rejected。

重点查是否存在不公平排除：candidate arena/sign/pre25/output storage不得从absolute peak扣除；native与
candidate必须使用相同input/state binding和allocator API。也请核对compile/admission不属于本阶段
timing，但live storage仍进入memory口径。

## 6. AC4 — 失败纪律

独立核对两次pre-formal失败：

- b2首次生成在写artifact前因错误冻结native hash fail closed，随后新commit重跑；
- b3首次formal在run0 candidate import前因覆盖`PYTHONPATH` fail closed，原子临时目录清理，随后
  `eeeb1bf`从run0重跑10 worker。

不得发现partial raw续跑、挑样、修改1.0x门槛或事后改变pair order。

## 7. AC5 — 回归和claim

- targeted R3=`10 passed`；
- full=`1595 passed, 3 skipped, 6 warnings`；
- mypy clean；pylint 10.00/10；DocOps lint PASS；
- `timing_recorded=false`、`performance_claimed=false`全链路保持；
- memory结果只限定frozen P-anchor single evaluation，不外推optimizer/query；
- 下一只开放R3-2A trajectory correctness，R3-2B timing继续关闭。

## 8. 建议命令

```text
python scripts/run_r3_compiled_p_alpha_vjp_artifact.py replay \
  --artifact artifacts/r3-structured-owner/r3-1b2-compiled-p-alpha-vjp-v1
python scripts/probe_r3_compiled_p_alpha_vjp_tamper.py \
  --artifact artifacts/r3-structured-owner/r3-1b2-compiled-p-alpha-vjp-v1
python scripts/run_r3_compiled_five_fresh_artifact.py \
  --artifact artifacts/r3-structured-owner/r3-1b3-compiled-five-fresh-v1 --replay
python scripts/probe_r3_compiled_five_fresh_tamper.py \
  --artifact artifacts/r3-structured-owner/r3-1b3-compiled-five-fresh-v1
pytest -q tests/test_r3_compiled_five_fresh_artifact.py \
  tests/test_r3_compiled_p_alpha_vjp.py \
  tests/test_r3_compiled_p_alpha_vjp_artifact.py \
  tests/test_r3_p_alpha_vjp_oracle.py
```

## 9. 输出格式

请报告：verdict、AC1–AC5逐项PASS/FAIL、blocker/major/minor/info、独立重算数字、不能现场复核项，
以及是否同意只开放R3-2A。不要把memory reduction写成latency或query speedup。
