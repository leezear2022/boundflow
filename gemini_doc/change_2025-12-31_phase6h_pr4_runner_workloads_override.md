# 变更记录：Phase 6H PR-4 补丁——runner 支持 workloads 覆盖（kick-the-tires 默认不变）

## 动机

Phase 6H 的一键 runner 需要同时满足两类需求：

- **Kick-the-tires ≤30min**：默认路径尽量短、稳定，因此默认只跑 `1d_relu`。
- **更像论文主结果的 workload suite**：可选加入小型非 toy MLP（例如 `mlp2d_2x16`），但不应强迫所有 AE 路径都变慢。

因此本补丁让 runner 支持通过参数/环境变量覆盖 workload 列表，同时保持默认行为不变。

## 本次改动

- 更新：`scripts/run_phase6h_artifact.sh`
  - 支持第二个参数 `WORKLOADS`（逗号分隔）。
  - 支持环境变量 `PHASE6H_WORKLOADS`。
  - 默认仍为 `1d_relu`（kick-the-tires）。
- 更新：`gemini_doc/ae_readme_phase6h.md`
  - 增加 “可选：扩展到非 toy workload” 的运行说明。

## 如何验证

```bash
# 默认（kick-the-tires）
bash scripts/run_phase6h_artifact.sh /tmp/phase6h_artifact_run

# 扩展到非 toy
bash scripts/run_phase6h_artifact.sh /tmp/phase6h_artifact_run "1d_relu,mlp2d_2x16"
```

