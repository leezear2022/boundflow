# FSG4/B3-C Device Commit 外部审计交接

日期：2026-08-14
审计目标：判断`VALIDATED-B3-C-COUNTERS`是否可关闭；不得升级为performance claim。

## 1. 审计对象

- branch：`feat/rvir-v4-production-state-ownership-v1`；
- implementation source：`72bec5ee1bdabfdefbf51201ac49395489eeef65`；
- closure commit：以本文件所在HEAD为准；
- PR：#60；
- artifact：`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3c-v1/`；
- tamper：`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3c-v1-tamper-report.json`。

## 2. 需要独立复核的验收项

### AC1：真实GPU与source provenance

- manifest source必须为`72bec5e…ef65`；
- code revision必须覆盖B3-C两个新runtime文件及完整B2/B3-A/B依赖；
- model/property及αβ-CROWN、auto_LiRPA、VNN-COMP identity不得漂移；
- worker environment必须admitted，provider/fallback必须全零。

### AC2：物理激活

从`events.jsonl`独立重算，不采信report聚合：

- candidate/commit/backup/copy=`12/12/12/12`；
- timed candidate D2H=`0`；
- template compile/hit=`1/1`、module move=`0`、scope=`1`；
- optimizer=`10/9`、snapshots=`0`、forward=`4`、KFSB=`3/3`。

### AC3：headline与audit隔离

- assembly `headline_content_digest_count=0`；
- query end event与wall time必须在`finalize_post_query_audit()`之前结束；
- audit必须在CUDA sync之后运行，`post_query_audit_ns>0`且excluded；
- 24次GPU content hash应由12个candidate + 12个committed digest构成；
- audit hash、transaction version和commit hash必须交叉绑定。

### AC4：transaction fail-closed

独立阅读/运行测试，至少覆盖：NaN、stale tensor version、wrong inventory/placement/alias、五个空beta
`data_ptr()==0`但不别名、第五次copy失败、host写入失败、opaque discarded host对象、receipt完整重签名
篡改。失败时12个tensor和host pre-image必须恢复。

### AC5：语义与artifact replay

- 与FSG3 v5六个冻结B2 control逐项比较，不只比较一个历史truth；
- replay必须从source commit读取code revision；
- 六类outer-resigned攻击必须6/6拒绝；
- full test必须为`1279 passed, 3 skipped`，并核对skip理由。

### AC6：claim边界

- 只能关闭B3-C mechanism/correctness/counter claim；
- 不得从单次run的query/core wall外推speedup；
- 5 fresh B2/B3 pairs与36-process正式计时仍未完成；
- B4—B7仍关闭。

## 3. 建议命令

```bash
conda activate boundflow
python scripts/run_fsg4_b3_counter_diagnostic.py replay \
  --artifact-dir artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3c-v1
python scripts/probe_fsg4_b3_counter_artifact_tamper.py \
  --artifact-dir artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3c-v1 \
  --report /tmp/fsg4-b3c-audit-tamper.json
pytest -q tests/test_fsg4_b3_device_atomic_commit.py \
  tests/test_fsg4_b3_device_live_return.py \
  tests/test_fsg4_b3c_device_commit_artifact.py
pytest -q tests
```

## 4. 已知失败历史

首次source `a3ac761` run因递归序列化opaque host provider对象在mutation前fail closed，没有留下正式
artifact。审计应确认修复没有放松key inventory或retained三字段门禁。

## 5. 期望输出格式

请给出：总体verdict；AC1—AC6逐项PASS/FAIL及独立证据；blocker/major/minor/info findings；不可现场
审计边界；是否同意`VALIDATED-B3-C-COUNTERS`关闭；下一动作是否仍为5 fresh correctness pairs。
