# 变更记录：IR-5 路线封存与发布交接

> 日期：2026-08-03
> 分支：`feat/compiler-ir-stack-v1`
> closure parent：`5a29a8e`
> 判定：IR-5 final VALIDATED-NO-GO；当前 ASPLOS system-performance 路线封存

## 1. 封存对象

- IR-1—4 typed Bound/Plan/Task/Schedule/runtime validated-reduced 实现；
- IR-5C3 原始 fair architecture-held-out No-Go；
- IR-5D prepared execution remediation；
- IR-5F v2 protocol-invalid 根因与退役 identities；
- IR-5H fresh residual-final-v3 完整 artifact 与最终 No-Go。

最终 artifact：`artifacts/ir5/residual-final-v3-20260728`。manifest 绑定
`971a3175af3cadee7eff1138837354740dbff026`，integrity/semantic replay 已通过。

## 2. 最终门禁

- correctness / exact input identity / 8-context feasibility：PASS；
- Global p90 regret：`1.26160× > 1.20×`，FAIL；
- 双 workload compiler Pareto：gray 仅单一 frontier 点，FAIL；
- multi-budget switch：未出现；
- IR-6：不启动。

不得旋转 seed、重用 `7501/7502` 调参、删除公平 baseline 或把局部 backend 收益升级为
系统级 Planner claim。

## 3. 文档一致性修复

本轮把权威当前状态中仍写作“下一步补 IR-5 / 当前进入 IR-5 / 只允许 IR-5D”的残留指令
改为历史完成语义，并把总体计划/执行备忘录置顶更新到最终 No-Go。历史 change 文档保留
当时的下一步，不回写历史。

## 4. 外部审计命令

```bash
git checkout feat/compiler-ir-stack-v1
conda activate boundflow
python scripts/run_ir5_family_fair_artifact.py replay \
  --suite residual-final-v3 \
  --artifact-dir artifacts/ir5/residual-final-v3-20260728
python scripts/run_ir5_family_fair_artifact.py replay \
  --suite residual-final-v3 \
  --artifact-dir artifacts/ir5/residual-final-v3-20260728 --semantic
pytest -q tests
```

2026-08-03 本机复核结果：

- integrity replay：PASS；
- 全量回归：`445 passed, 37 skipped`，无失败；
- semantic replay：本轮未能现场复跑，原因是宿主机 NVIDIA 驱动不可通信，
  `torch.cuda.is_available() == False`；这不改写 2026-07-28 正式 CUDA artifact 已通过
  semantic replay 的历史证据，但属于本次发布审计的明确环境边界。

## 5. 后续研究准入条件

当前分支只做 closure，不继续性能调优。若另开路线，必须使用独立分支和新契约。推荐顺序：

1. 修复真实 ResNet whole-query bound equivalence；
2. activation-BaB 的真实 query coverage 从 `0/394` 提升为 typed IR 可执行覆盖；
3. 将真实 αβ-CROWN query 映射到 Plan/Task/Schedule IR；
4. 先冻结 correctness artifact，再决定是否存在新的性能研究问题。

这是一条新的 correctness/integration 研究路线，不是 IR-5 的续跑。
