# FSG3 B0/B1/B2 Same-Solver 正式基线 — 外部审计交接

## 1. 审计目标

请独立判断FSG3是否可按`VALIDATED-FSG3-B0-B1-B2-BASELINE`关闭，以及B2是否只能分类为
`MEASURED-B2-SLOWER`。不要采信本文数字；应从冻结raw artifact与代码重新计算。

本轮不是“BoundFlow加速成功”审计。`performance_claimed=false`必须保持，B3—B7、31-node queue、
complete-query、TTV、multi-workload和ASPLOS-ready均未关闭。

## 2. 冻结输入

- branch：`feat/rvir-v4-production-state-ownership-v1`；
- Draft PR：`#60`，base=`main`；
- formal source：`a4ee2910f4039981338fb6d8688ac4af18508b73`；
- artifact：`artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5/`；
- tamper report：
  `artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5-tamper-report.json`；
- runner：`scripts/run_fsg3_same_solver_experiment.py`；
- worker：`scripts/run_fsg3_same_solver_worker.py`；
- schema：`boundflow/runtime/fsg3_same_solver_timing.py`；
- artifact tests：`tests/test_fsg3_same_solver_artifact.py`；
- closure说明：`gemini_doc/change_2026-08-14_fsg3_same_solver_formal_baseline.md`。

预期summary/manifest hash：

```text
df852590d99be09962c1287e7166b421edb260416403a3c91545dca6e2e1318e
9089e2019eb5e98cac228151cb061c0f6aceefa0ad6c6b3e298584bcede21e85
```

## 3. 必做命令

```bash
conda activate boundflow
source env.sh

python scripts/run_fsg3_same_solver_experiment.py replay \
  --artifact-dir artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5

python scripts/probe_fsg3_same_solver_artifact_tamper.py \
  --artifact-dir artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5 \
  --report /tmp/fsg3-tamper-report.json

pytest -q tests/test_fsg3_same_solver_timing.py \
  tests/test_fsg3_same_solver_artifact.py
```

审计方应另写一个最小独立解析器读取JSON/JSONL；不要只调用repo replay。

## 4. Acceptance Criteria

### AC1 — provenance与覆盖

- manifest schema为`boundflow.fsg3-same-solver-artifact/v4`且source commit exact；
- 六个B0/B1/B2全排列block顺序exact；
- 每配置6 control+6 profile，总计36个fresh worker；
- summary hash与manifest hash独立重算一致，所有manifest文件SHA256一致。

### AC2 — 环境与测量可审计

- 36/36 `environment.admitted=true`且runtime identity count为1；
- 独立thermal/hardware slowdown均未激活；严格镜像的SW power/thermal counter按coupled alias处理，
  不得误称独立thermal；
- 18个profile的closure error `<=1%`、residual `<=3%`；
- B0/B1/B2 profile/control ratio全部`<=1.05`。

### AC3 — correctness与物理路径

- 全部semantic/state/branch/queue/termination门禁通过，failure rows为空；
- 每个B0的typed/provider/fallback为`0 / 1,14,3 / 0`；
- 每个B1为`1 / 1,14,3 / 0`；
- 每个B2为`1 / 0,0,0 / 0`；
- B2确为whole-call reference replacement，不是original callback或post-hoc IR。

### AC4 — paired统计

所有ratio方向必须是`B0/candidate`。从control raw独立重算并至少核对：

- B1 query wall geomean=`0.9956565794571265`；
- B2 query wall geomean=`0.9083995539523697`；
- B2 core wall geomean=`0.5167670145223869`；
- B1/B2 allocated与reserved ratio均为`1.0`；
- B2 compile break-even=`not_reachable`。

审计结论必须明确：这些数值说明当前B2较慢，不是speedup。

### AC5 — attribution与传播边界

从profile raw独立重算B2 core wall share：optimizer约44.0%、atomic commit约24.7%、KFSB约16.7%、
typed pre-state约10.7%、backward约3.7%。不得把单个selected-CROWN/backward区域外推为全栈上限；
也不得把B2较慢外推为B3—B7无潜力。

### AC6 — tamper fail closed

8类攻击必须在修改payload并同步更新manifest文件digest与manifest hash后仍被拒绝。审计报告必须称为
“outer-resigned”，除非审计方另行证明所有内部语义副本也同步更新。

### AC7 — regression与claim纪律

- artifact targeted=`33 passed`；Black、mypy、Pylint=`10.00/10`；全量=
  `1233 passed, 3 skipped`；
- raw artifact保留`performance_claimed=false`；
- 权威claims/status/计划都将FSG3写为baseline关闭、B2较慢、FSG4/B3下一门禁；
- ASPLOS-ready仍为NO，B7最终门槛未测试。

## 5. 允许的结论

- `APPROVE`：AC1—AC7全过；
- `APPROVE-WITH-MINOR`：只有不改变数字、语义、门禁或复现性的轻微问题；
- `REQUEST-CHANGES`：任一覆盖、provenance、正确性、环境、统计方向、tamper或claim传播问题失败。

## 6. 审计报告格式

请逐项给出AC1—AC7的PASS/FAIL、独立命令与独立重算结果；findings按blocker/major/minor/info分级；
最后给出是否允许启动FSG4/B3。无法现场重跑36个GPU进程不是自动失败，但必须清楚区分“冻结artifact
replay”与“从source重新生成”。
