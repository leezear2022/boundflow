# RVIR 审计后加固复审关闭记录

> 日期：2026-08-04
> 基线：`main@ffe0071`
> 分支：`docs/close-rvir-post-hardening-audit`
> DocOps task：`rvir-post-hardening-20260803`

## 1. 结论

外部审计 round 1 已通过 DocOps 正式提交，verdict 为 `approve`：

- AC1—AC6 全部 PASS；
- 原审计 F1—F5 全部 closed；
- 无 blocker、major 或 minor finding；
- 仅保留一条 info：Phase6H 在 override 与 `CONDA_PREFIX` 均缺失时回退 PATH 中的
  `python`，属于已文档化行为，可选增加 `import boundflow` 自检改善报错；
- 无 claim drift，RVIR 保持 VALIDATED-REDUCED，IR-5 保持 VALIDATED-NO-GO；
- 不形成 performance、CUDA、fused execution、完整 verifier E2E 或 ASPLOS-ready 主张。

executor 已运行 `dol exchange close`，task 状态为：

```text
approved → closed
resolution: approved
approved_round: 1
```

## 2. Git 固定的审计材料

本次提交统一固定：

- `r001/audit.md` 与 `audit.json`：DocOps 正式审计文档；
- `r001/audit_report_full.md`：外部审计的完整命令与逐项证据附件；
- `closure.md` 与 `closure.json`：executor 正式关闭记录；
- `state.json`：`status=closed`、`rev=6`；
- `.docops/ev.jsonl` 与 `.docops/s.md`：本地操作和当前状态。

`audit.json.md_sha256` 与 `audit.md` 实测一致：
`14ff5bd15d179155c9a4f46e11c84d773ea9bda4e53218fa1701ae4e7d4be0a9`。
完整版附件不属于 DocOps audit JSON 的内嵌 digest 集合，因此由本 Git commit 固定其内容。

## 3. executor 复核

- `dol exchange validate rvir-post-hardening-20260803`：PASS；
- `dol exchange validate rvir-20260803`：PASS；
- `dol lint --soft`：PASS；
- AC1 专项复跑：`6 passed`；
- AC4 独立重算：377 query / 377 record、root/parent 30/347、parent-before-child、
  completed 与 manifest digest 全部成立；
- AC5 临时目录的三个 commit、两个输入 SHA256、12 个冻结字段与 8 个 tensor digest 均匹配；
- 旧 exchange 与 RVIR v1 artifact 在审计范围内零改动。

审计后 VNN-COMP checkout 会生成未跟踪 `.vnnlib.compiled` cache；这不属于 tracked 输入修改，
也不进入结果身份。

## 4. 后续

RVIR correctness/integration 路线及其审计后加固均已关闭。下一步不自动启动性能路线；应重新
选择 ASPLOS workstream。若未来开启 performance/CUDA，必须新建 task，冻结公平 lower-only
输出合同并取得 fresh GPU evidence，不能复用本轮 correctness artifact 升级 claim。
