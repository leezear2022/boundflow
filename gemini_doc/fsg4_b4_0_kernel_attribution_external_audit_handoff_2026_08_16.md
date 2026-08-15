# FSG4/B4-0 Kernel Attribution 外部审计交接

## 1. 审计目标

请不要采信closure/summary中的数字；从formal raw独立判断B4-0是否可以关闭为
`VALIDATED-B4-0-OPPORTUNITY`，以及是否只应开放B4-A cumulative candidate。

审计对象：

- source：`66154e485594e8a84ad1ce04d701d8543c1a7335`；
- artifact：`artifacts/fsg4-b4-kernel-attribution/resnet2b-prop0-v1/`；
- runner：`scripts/run_fsg4_b4_kernel_attribution.py`；
- typed schema：`boundflow/runtime/fsg4_b4_kernel_attribution.py`；
- prereg：`gemini_doc/BOUNDFLOW_FSG4_B4_CUMULATIVE_CUDA_TIR_FUSION_PLAN_2026_08_16.md`；
- internal closure：`gemini_doc/change_2026-08-16_fsg4_b4_0_kernel_attribution_closure.md`。

## 2. Acceptance criteria

### AC1 — Source/protocol/raw identity

- source/code revision、B3 manifest、模型/性质及外部仓库commit独立重算一致；
- manifest、compressed file、uncompressed JSONL、canonical raw、worker/summary多层hash一致；
- artifact零本机绝对路径泄漏。

### AC2 — Fresh semantic pair

- control/profile均为fresh B3 worker，顺序=`control,profile`；
- discrete/sign exact；lower allclose使用protocol绑定的B3冻结`atol=rtol=2e-4`；
- profile时间只作扰动披露，`performance_claimed=false`。

### AC3 — 14-call phase closure

- 从raw独立重建14个CROWN ordinal和4个forward ordinal；
- 所有真实CUDA kernel均进入closure；correlation/temporal/unattributed计数与raw一致；
- CUDA user annotation不得误算为kernel；stream、shape、phase parent与allocation delta可解析。

### AC4 — Opportunity计算

- 从raw独立聚合optimizer 10、terminal export 1、KFSB 3的kernel/materialization ledger；
- 从B3 formal raw独立复算CROWN14/whole-core share比值；
- 判断B4-A是否满足“消除一个完整重复CROWN call”，B4-B是否满足`>=5% B3 core`；
- 明确kernel-sum和allocation delta不得升级为wall speedup或memory saving。

### AC5 — Replay/tamper

- root replay逐字段重建summary；
- 独立重跑9类outer-resigned probe，或检查其确实更新内外层digest后仍被semantic replay拒绝。

### AC6 — Regression/static/DocOps

- B4 targeted=`15 passed`；B3/B4相关=`54 passed`；full=`1329 passed, 3 skipped`；
- Black/Mypy/Pylint/diff check和`dol exchange validate`/`dol lint --soft`通过。

### AC7 — Claim边界

- B4-0只允许attribution/opportunity claim，无B4 speedup、B0 parity、memory saving或ASPLOS-ready；
- approve后只开放B4-A，B4-B可设计但不得合并执行，B4-C/D与B5—B7继续关闭。

## 3. 建议独立命令

```bash
python scripts/run_fsg4_b4_kernel_attribution.py replay \
  --artifact-dir artifacts/fsg4-b4-kernel-attribution/resnet2b-prop0-v1

python scripts/run_fsg4_b4_kernel_attribution.py tamper \
  --artifact-dir artifacts/fsg4-b4-kernel-attribution/resnet2b-prop0-v1

python -m pytest -q \
  tests/test_fsg4_b4_kernel_attribution.py \
  tests/test_fsg4_b4_kernel_attribution_runner.py
```

独立raw解析应优先使用Python标准库`gzip/json/hashlib`，避免调用被审计方summary helper。

## 4. 审计输出格式

请按AC1—AC7逐项给`PASS/FAIL`、证据与独立重算数字；findings分`blocker/major/minor/info`。最终verdict
只能为`approve`、`approve-with-findings`或`reject`。若approve，请明确是否同意关闭
`VALIDATED-B4-0-OPPORTUNITY`并只开放B4-A。
