# FSG4/B3 36-Process 正式计时外部审计交接

日期：2026-08-14

审计目标：独立判断`VALIDATED-REDUCED-B3`是否成立，以及是否允许以 B3 为累计基线进入 B4。

## 1. 审计对象

- branch：`feat/rvir-v4-production-state-ownership-v1`；
- PR：#60；
- artifact source：`36e9069ca4f21183c9b36d74024de0ca8b20f59c`；
- artifact：`artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1/`；
- tamper：`artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1-tamper-report.json`；
- plan：`gemini_doc/BOUNDFLOW_FSG4_B3_IR_GRAPH_PLAN_SCHEDULE_REUSE_PLAN_2026_08_14.md`；
- closure：`gemini_doc/change_2026-08-14_fsg4_b3_formal_timing_closure.md`；
- frozen test：`tests/test_fsg4_b3_same_solver_artifact.py`。

## 2. Acceptance Criteria

### AC1：Source、输入与正式协议

- source必须 exact 为`36e9069…f59c`，code revision必须覆盖runner、replay、tamper和B3完整依赖；
- protocol必须绑定前一阶段five-fresh manifest internal/file hash；
- model/property、αβ-CROWN、VNN-COMP、解释器、runtime和GPU identity必须可从raw独立核对；
- artifact所有文本不得泄漏本机绝对路径。

### AC2：36个 Fresh Worker 与固定顺序

- 必须有六个 B0/B2/B3全排列block；每个配置在每个block各一个control和profile，总数36；
- B0/B2/B3均必须为`6 control + 6 profile`；
- 每个worker必须是独立subprocess，不能在一个CUDA context内循环；
- raw-first/resume只能接受完整、source-bound、可重放worker；不得从部分结果补写formal manifest。

### AC3：Correctness 与物理 Activation

- 从raw重新验证所有direct semantic pair和environment gate，不能只读summary布尔值；
- B0必须为original provider；B2/B3 provider core/compute/update/fallback必须全零；
- 每个B3 worker必须有prepared template、PlanInstance、terminal Schedule、assembly、commit和post-query
  audit直接receipt，headline digest与candidate D2H为零；
- B2/B3 profile counter必须分别保持snapshots/forward/D2H=`10/5/12`与`0/4/0`，optimizer=`10/9`；
- control必须没有详细counter instrumentation。

### AC4：Measurement Admission

- 36/36 environment admitted，runtime identity count=`1`；
- 18/18 profile closure通过，最大closure/residual需独立重算为约`0.0025104990`；
- B0/B2/B3 profile/control query perturbation最大值均须`<=1.05`；
- headline只使用control配对，profile只用于归因。

### AC5：独立 Ratio 与分类重算

请直接从`worker_runs.jsonl`按相同block的control pair重算几何平均，不采信`summary.json`数字：

- B2/B3 core wall应为`1.0716174805930418x`；
- B2/B3 query wall应为`1.0066228954759742x`；
- B0/B3 query wall应为`0.9100012637918488x`；
- 六个B2/B3 core pair最小值应为`1.0635877032562384x`；
- 根据冻结阈值应恰好分类`VALIDATED-REDUCED-B3`，不能升级full B3，也不能误判NO-GO；
- peak allocated/reserved没有实质改善，不得产生memory claim。

### AC6：Replay、Tamper 与冻结证据

- root replay必须从36个raw run重建sequence、semantics、activation、closure、ratio与decision；
- 独立运行十类outer-resigned attack，必须`10/10 rejected`；
- 特别确认修改raw latency/semantic/counter并同步重签外层digest后仍由派生重算拒绝；
- frozen test必须绑定source、固定顺序、direct activation、内部/file hash和tamper report；
- targeted应为`114 passed`，full应为`1314 passed, 3 skipped`，并核对skip理由。

### AC7：Claim 边界与下游门禁

- 只允许`VALIDATED-REDUCED-B3`：B3相对B2 core改善且query不退化，但仍未回到B0 query parity；
- artifact的`performance_claimed=false`必须保持；不能声称BoundFlow已经快于auto_LiRPA；
- 单workload、固定solver prefix、单RTX 4060的限制必须保留；
- 审计通过后只允许以B3为B4累计候选；B5—B7及最终`1.20x queue / 1.15x complete-query`仍关闭。

## 3. 建议命令

```bash
conda activate boundflow
python scripts/run_fsg4_b3_same_solver_experiment.py replay \
  --artifact-dir artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1
python scripts/probe_fsg4_b3_same_solver_artifact_tamper.py \
  --artifact-dir artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1 \
  --report /tmp/fsg4-b3-formal-audit-tamper.json
pytest -q tests/test_fsg4_b3_same_solver_artifact.py
pytest -q tests/test_fsg3_same_solver*.py tests/test_fsg4_b3*.py
source env.sh
pytest -q tests
```

审计方还应写一个独立短脚本从`worker_runs.jsonl`按block重算AC5，不调用executor侧
`derive_fsg4_b3_timing_evidence()`。

## 4. 已知边界

- 正式 workload是ResNet2B property 0、固定一次solver prefix，不是complete solve/TTV实验；
- full回归的3个skip为一个TVM重复编译规避与两个独立冻结VNN-COMP checkout不可用测试，不影响本artifact
  自带的VNN-COMP input digest和36/36正式运行；
- 本轮不要求重新生成约16分钟的36-process artifact，但必须重放已冻结raw并独立重算；如要原地重跑，
  必须使用新目录，不能覆盖executor artifact。
- `logs/*.stdout.txt`/`stderr.txt`原样保存上游solver输出，部分行有上游行尾空格且已被manifest digest绑定；
  不得为通过格式检查改写日志。source/docs/JSON/JSONL仍应单独通过`git diff --check`。

## 5. 期望输出

请给出总体verdict；AC1—AC7逐项PASS/FAIL及独立证据；blocker/major/minor/info findings；不可现场审计
边界；是否同意`VALIDATED-REDUCED-B3`关闭；是否同意只开放B4 cumulative candidate。
