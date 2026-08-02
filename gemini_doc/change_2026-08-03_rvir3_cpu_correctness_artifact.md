# 变更记录：RVIR-3/4 CPU Correctness Artifact

> 日期：2026-08-03
> 分支：`feat/real-verifier-ir-integration-v1`
> Artifact：`artifacts/rvir/rvir-cpu-correctness-v1-20260803`

## 冻结内容

- `activation_calls.jsonl`：历史 PR-14A 的 394 个 activation-BaB query identity，以及逐
  query 重新生成的 BoundModule、PlanTemplate、PlanInstance、TaskModule、Schedule 五层
  stable hash；
- `online_execution.json`：adapter v2 的真实 CPU observer on/off 对照、377 次 exact-call
  dispatch、parent lineage 和 requested-output 统计；
- `resnet_semantics.json`：RVIR-1 的 6 组 external intermediate bounds、adaptive policy、
  lower allclose 和 sign evidence；
- `manifest.json`：所有文件 SHA256、历史来源 SHA256、coverage、semantics owner 与环境边界。

Artifact 内嵌 394 个完整 query identity，因此 replay 不依赖本机 ignored PR-14A 目录；fresh
process 会从每个 query 重新编译五层 IR，并逐行比较全部字段与 hash。

## 门禁结果

- 历史 activation admission：`394/394`，effective method 均为 αβ-CROWN；
- 历史 workload 分布：simple MLP 343、VNN-COMP ResNet 51；
- 真实在线 exact-call：query / dispatched / completed = `377/377/377`；
- 在线 activation：343；parent links：347；显式 lower-only：377；
- observer on/off：status 一致、visited domains `380/380`、final lower 均为
  `tensor(-0.18902308)`；
- ResNet initial-CROWN：lower max diff `3.09944e-6`、sign `9/9`；
- artifact fresh-process replay：PASS。

## 必须保留的限制

- 历史 adapter v1 的 394 行全部缺少精确 split tensor values、requested polarity 和 parent
  lineage；artifact 逐行写入
  `split_state_values_unresolved`、
  `legacy_requested_bound_polarity_unresolved_assumed_both`、
  `legacy_parent_lineage_not_captured`，不允许外部审计将其解释为完整 identity；
- adapter v2 的真实在线证据补齐 requested polarity 与 parent lineage，但只覆盖当前官方
  simple MLP CPU workload；
- external verifier 继续拥有 α/β/split 算法与 termination；fused replacement coverage 仍为
  `0/394`；
- 本机 CUDA 不可用，且 typed hash/validation 有明显开销；本 artifact 明确
  `performance_claimed=false`。

## 复核命令

```bash
conda run -n boundflow python scripts/run_real_verifier_ir_artifact.py replay \
  --artifact-dir artifacts/rvir/rvir-cpu-correctness-v1-20260803
```
