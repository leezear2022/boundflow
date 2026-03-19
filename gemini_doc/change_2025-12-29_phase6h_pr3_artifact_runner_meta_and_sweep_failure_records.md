# 变更记录：Phase 6H PR-3（AE/论文工件准备）——一键 runner + meta 补全 + sweep 失败记录

## 动机

在 6H PR-1/PR-2 之后，系统收益已经能通过 JSONL→CSV/fig 复现；PR-3 的目标是进一步贴近 AE/复现的“交付形态”：

- 提供“一条命令跑主结果”的 runner（减少沟通成本）；
- bench meta 补全 OS/Python 信息（提高可复现审计性）；
- sweep 遇到非 0 退出码时，不静默失败：把失败作为 JSONL 记录写出，并可选 fail-fast。

## 本次改动

- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - `meta` 增加：
    - `python_version`
    - `platform`（`platform.uname()` 的可序列化信息）

- 更新：`scripts/sweep_phase6h_e2e.py`
  - 增加 `--fail-fast`：遇到失败立即终止。
  - 默认行为：失败不丢失，追加写入 JSONL 一行 `{meta.run_status="error", error, returncode, stderr_tail, stdout_tail}`，并在结束时返回非 0（方便 CI/自动化捕获）。

- 更新：`scripts/report_phase6h_e2e.py`
  - `summary.md` 增加 “失败运行（run_status=error）” 区块（方便快速定位哪组配置跑挂）。

- 新增：`scripts/run_phase6h_artifact.sh`
  - 一键执行：`sweep → report → plot`，输出到 `artifact_out/phase6h_<timestamp>/`。

## 如何验证

```bash
# 运行一键 runner（默认输出到 artifact_out/...）
bash scripts/run_phase6h_artifact.sh
```

