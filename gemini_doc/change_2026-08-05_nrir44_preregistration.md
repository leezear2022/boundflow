# NRIR-44 Root-Projection Floor Schedule v1 预注册记录

## 起因

NRIR-43 证明更大的 CPU domain batch 会退化。进一步分解 NRIR-42 floor 后发现，约 21.77 秒中
9 条 deep objective queries 占约 13.88 秒，而 downstream ranking 只读取每条 root lower；随后
selected top-2 又各自运行完整 31-node production queue。这里存在可由 Plan/Schedule IR 描述的
result-liveness 与重复工作问题。

## 路线前探针

保持同一模型、property、objective refinements、optimizer 与 root input，单独将 9 条 query 配成
n1d0：合计 `0.789371 s`，9/9 root lower/upper/branch 与 n31d4 root exact。该单次 probe 只用于
选择路线，不作为正式性能结论。

## 预注册

- 唯一变量：ranking floor `9×n31d4 → 9×n1d0`；
- Phase A：root/rank/selected exact、evaluations `279→9`、floor `<=11 s`、ratio `<=0.50`；
- Phase B：NRIR-42 production exact、whole `<=48 s`、ratio `<=0.82`；
- 一般 complete verifier 不自动启用；非 top-2 深层证明机会被保守降为 unknown；
- 当前无实现、artifact 或新 claim，ASPLOS-ready 仍为 NO。
