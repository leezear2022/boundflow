# FSG4/B3-B Terminal Schedule 外部审计交接

日期：2026-08-14
执行方结论：`VALIDATED-B3-B-COUNTERS`，非performance claim

## 审计对象

- source：`42df2dcae2d5c5a10f27ab707d8d7aff7686d15e`；
- branch/PR：`feat/rvir-v4-production-state-ownership-v1` / draft PR #60；
- artifact：`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3b-v1/`；
- tamper：`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3b-v1-tamper-report.json`；
- implementation：`boundflow/runtime/fsg4_b3_terminal_optimizer_schedule.py`、
  `boundflow/runtime/rvir_v4_native_backward_export.py`、`scripts/run_rvir_v4_live_return_capture.py`；
- frozen tests：`tests/test_fsg4_b3b_terminal_schedule_artifact.py`。

## 必须独立复核

1. Schedule是否真实包含10 evaluation/9 update、双学习率及decay，而非仅元数据；
2. production executor是否不构造10份step对象，只保存terminal lower/α/β；formal trace是否保持；
3. forward trace是否绑定scope/graph/split和完整inventory，backward是否真实跳过一次forward build；
4. B2/B3-A feature-off路径是否不变，B3-B是否必须显式prepared core + terminal schedule；
5. 5157条event能否独立重算snapshots=`0`、forward=`4`及全部保持项；
6. 六个B2 control语义、provider/fallback、artifact digest/code revision/replay是否闭环；
7. 6类outer-resigned攻击是否全部被拒绝，report-only攻击是否不再是B3-B空变换；
8. 文档是否保留单fresh、无timing/speedup、完整B3仍需5 fresh pair的限制。

## 建议命令

```bash
conda activate boundflow
source env.sh
python scripts/run_fsg4_b3_counter_diagnostic.py replay \
  --artifact-dir artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3b-v1
python scripts/probe_fsg4_b3_counter_artifact_tamper.py \
  --artifact-dir artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3b-v1 \
  --report /tmp/fsg4-b3b-tamper-audit.json
pytest -q tests/test_fsg4_b3_terminal_optimizer_schedule.py \
  tests/test_fsg4_b3b_terminal_schedule_artifact.py \
  tests/test_fsg4_b3_explicit_counters.py
```

请给`APPROVE / APPROVE-WITH-MINOR / REQUEST-CHANGES`，按blocker/major/minor/info列finding，并明确区分
B3-B mechanism/correctness与尚未开始的正式B3 timing。
