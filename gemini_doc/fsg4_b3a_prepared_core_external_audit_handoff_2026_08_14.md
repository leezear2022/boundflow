# FSG4/B3-A Prepared Core 外部审计交接

日期：2026-08-14
执行方结论：`VALIDATED-B3-A-COUNTERS`，非performance claim

## 审计对象

- source：`c7851c8bae1bc943aa9e3d458e5105deafc553f1`；
- branch/PR：`feat/rvir-v4-production-state-ownership-v1` / draft PR #60；
- artifact：`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3a-v1/`；
- tamper report：`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3a-v1-tamper-report.json`；
- implementation：`boundflow/runtime/fsg4_b3_prepared_core.py`、
  `scripts/run_rvir_v4_live_return_capture.py`、`boundflow/runtime/rvir_v4_native_optimizer.py`；
- frozen tests：`tests/test_fsg4_b3a_prepared_core_artifact.py`。

## 必须独立复核

1. template是否真实绑定graph/parameter、topology、device/dtype、input/objective shape、policy contract、
   12 mutable paths与binding placement；
2. dynamic instance是否重新绑定snapshot/mapping/input/objective/bounds/split/α/β/policy，且只构造一次scope；
3. B2 feature-off路径是否未改变，prepared路径是否必须显式cache/hash pair；
4. 5157条event能否独立重算出template compile/hit=`1/1`、module move=`0`、scope=`1`，其余冻结
   counter是否保持；
5. worker是否与FSG3 v5六个B2 control语义一致，provider/fallback是否全零；
6. artifact code revision、file digest、snapshot/report/manifest hash和replay是否闭环；
7. 六类outer-resigned攻击是否确实被语义/counter/code门禁拒绝；
8. 文档是否始终保留“单fresh、无timing/speedup、完整B3仍需5 fresh pair”的限制。

## 建议命令

```bash
conda activate boundflow
source env.sh
python scripts/run_fsg4_b3_counter_diagnostic.py replay \
  --artifact-dir artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3a-v1
python scripts/probe_fsg4_b3_counter_artifact_tamper.py \
  --artifact-dir artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3a-v1 \
  --report /tmp/fsg4-b3a-tamper-audit.json
pytest -q tests/test_fsg4_b3_prepared_core.py \
  tests/test_fsg4_b3a_prepared_core_artifact.py \
  tests/test_fsg4_b3_explicit_counters.py \
  tests/test_fsg4_b3_counter_artifact.py
```

## 期望审计输出

给出`APPROVE / APPROVE-WITH-MINOR / REQUEST-CHANGES`，逐条回答上述8项，并按blocker/major/minor/info
列finding。特别区分“B3-A physical/correctness已验证”与“B3 timing尚未开始”。
