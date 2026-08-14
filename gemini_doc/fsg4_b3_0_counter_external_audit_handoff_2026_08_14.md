# FSG4/B3-0 显式 Counter 正式关闭 — 外部审计交接

## 审计结论目标

请独立判断B3-0是否可关闭为`VALIDATED-B2-COUNTERS`。不要审计或批准B3 speedup：本轮明确
`diagnostic_timing_claimed=false/performance_claimed=false`。

## 冻结对象

- source：`419536126504e2666a5db14681668b7d1add166a`；
- artifact：`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b2-v1/`；
- manifest hash：`ccf15ee17cb1ee74b95984a203cb4893e52d70becbc3ba2d3db70618490bb376`；
- report hash：`4304ffe87ce09c6e14ff633ae72f469b6b1fb7c60d297179e74176a3a41ad68e`；
- tamper report：`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b2-v1-tamper-report.json`；
- tamper report hash：`f6392fa609c02d043b2397e36e54e52124630aa93fe51679892058efff644d1d`。

## 必核事实

1. raw worker是否为B2/control、CUDA、environment admitted，provider core/compute/update和fallback全零；
2. 4625条event是否按ordinal连续、report counter是否能从journal独立重算；
3. 固定counter是否为template=`1/0`、module/scope=`1/2`、optimizer=`10/9/10 snapshots`、forward=`5`、
   KFSB=`3/3`、D2H/commit/backup/copy=`12/12/12/12`；
4. 观察型counter `4417/45 tensor hash`、`84 validate`、`10 stable hash`是否由journal导出，而非手填；
5. FSG3 v5 manifest/file digest/36-run顺序是否先验证，当前semantic是否对六个B2 control全部通过；
6. 六类outer-resigned attack是否真的同步更新外层hash后仍分别被counter、journal、semantic、provider和
   code provenance门禁拒绝；
7. 两次失败尝试（α D2H漏计、Python 3.11/3.12 geomean表示差异）是否如实记录且没有降低门槛；
8. artifact、claims、status是否始终拒绝performance claim，并只开放B3-A。

## 建议命令

```bash
conda activate boundflow
source env.sh
python scripts/run_fsg4_b3_counter_diagnostic.py replay \
  --artifact-dir artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b2-v1
python scripts/probe_fsg4_b3_counter_artifact_tamper.py \
  --artifact-dir artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b2-v1 \
  --report /tmp/fsg4-b3-counter-tamper-audit.json
pytest -q tests/test_fsg4_b3_explicit_counters.py \
  tests/test_fsg4_b3_counter_artifact.py \
  tests/test_fsg3_same_solver_worker.py \
  tests/test_fsg3_same_solver_artifact.py
```

## Verdict 格式

请给`APPROVE / APPROVE-WITH-MINOR / REQUEST-CHANGES`，按blocker/major/minor/info列finding，并明确回答：

- B3-0 counter baseline是否可信；
- 是否允许从B3-0进入B3-A；
- 哪些counter是B3-A/B/C的硬消除目标；
- 是否存在任何被误写成speedup的诊断值。
