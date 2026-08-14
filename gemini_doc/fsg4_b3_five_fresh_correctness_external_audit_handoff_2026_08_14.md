# FSG4/B3 五组 Fresh Correctness 外部审计交接

日期：2026-08-14
审计目标：判断`VALIDATED-B3-FIVE-FRESH-CORRECTNESS`是否成立，以及是否可以只开放正式B3计时。

## 1. 审计对象

- branch：`feat/rvir-v4-production-state-ownership-v1`；
- source：`75dfd8103e8e3dfe824a63e15c2222f8742e28c1`；
- PR：#60；
- artifact：`artifacts/fsg4-b3-correctness-pairs/resnet2b-prop0-v1/`；
- tamper：`artifacts/fsg4-b3-correctness-pairs/resnet2b-prop0-v1-tamper-report.json`；
- plan：`gemini_doc/BOUNDFLOW_FSG4_B3_FIVE_FRESH_CORRECTNESS_PLAN_2026_08_14.md`；
- closure：`gemini_doc/change_2026-08-14_fsg4_b3_five_fresh_correctness_closure.md`。

## 2. Acceptance Criteria

### AC1：Source、输入与顺序

- source必须exact为`75dfd81…28c1`，code revision覆盖runner和B3-C完整依赖；
- model/property digest必须为`791aa24d…4a6d`/`89edf066…9ff`；
- 顺序必须exact为`B2→B3-C, B3-C→B2, B2→B3-C, B3-C→B2, B2→B3-C`；
- 必须是10个独立diagnostic subprocess，不是在一个CUDA context里循环。

### AC2：Raw-first与replay

- 独立统计必须有10个nested manifest、B2/B3-C各5个；
- root manifest必须绑定protocol、report和全部40个raw files；
- root replay必须逐个调用nested diagnostic replay并从raw重建pair，不得只比较root report hash；
- `--resume`只能接受完整且source一致、可replay的已有worker。

### AC3：五组直接语义

从每组B2/B3-C的raw worker重新比较：

- 离散state、branch、queue、termination exact；
- lower/finite upper满足`atol=rtol=2e-4`，B3-C不得optimistic；
- 5/5 pair的`semantic_failures=[]`；
- source/protocol/runtime/GPU identity一致且environment admitted。

### AC4：Physical activation与audit

- 每个B2 worker：4625 events、snapshots/forward/D2H=`10/5/12`；
- 每个B3-C worker：1484 events、template hit=`1`、module move/snapshots/D2H=`0/0/0`、scope=`1`、
  forward=`4`、optimizer=`10/9`、KFSB=`3/3`；
- 所有worker provider core/compute/update/fallback=`0/0/0/0`；
- 每个B3-C worker headline digest=`0`，post-query audit存在、excluded且绑定assembly/commit/audit。

### AC5：Tamper与冻结测试

- 独立运行7类outer-resigned攻击，必须`7/7 rejected`；
- 特别确认nested semantic和counter payload在同步更新外层digest后仍由重算拒绝；
- frozen artifact test必须校验source、root internal hash、5组顺序、10个raw counter和tamper report；
- full应为`1289 passed, 3 skipped`，并核对skip理由。

### AC6：Claim边界

- 只能关闭`VALIDATED-B3-FIVE-FRESH-CORRECTNESS`；
- 五组artifact必须仍写`timing_admitted=false/performance_claimed=false`；
- 不得从raw wall字段计算或主张任何speedup；
- 通过后只允许开放36-process B0/B2/B3正式计时，B4—B7继续关闭。

## 3. 建议命令

```bash
conda activate boundflow
python scripts/run_fsg4_b3_correctness_pairs.py replay \
  --artifact-dir artifacts/fsg4-b3-correctness-pairs/resnet2b-prop0-v1
python scripts/probe_fsg4_b3_correctness_pairs_tamper.py \
  --artifact-dir artifacts/fsg4-b3-correctness-pairs/resnet2b-prop0-v1 \
  --report /tmp/fsg4-b3-five-fresh-audit-tamper.json
pytest -q tests/test_fsg4_b3_correctness_pairs.py \
  tests/test_fsg4_b3_correctness_pairs_artifact.py
source env.sh
pytest -q tests
```

## 4. 已知环境事件

不加载`env.sh`且覆盖`PYTHONPATH`时，旧PR12测试会因无法导入vendored TVM在collection失败。正式全量验证
加载了`env.sh`，结果为`1289 passed, 3 skipped`。审计不得把该环境事件误判为artifact semantic失败，
也不得因此跳过完整环境复现。

## 5. 期望输出

请给出总体verdict；AC1—AC6逐项PASS/FAIL及独立证据；blocker/major/minor/info findings；不可现场审计
边界；是否同意正确性门禁关闭；是否同意只开放36-process正式B3计时。
