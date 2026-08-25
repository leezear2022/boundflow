# BoundFlow MR3-0 Provider Hook Feasibility 计划

> 日期：2026-08-26  
> 性质：MR3 implementation preflight；只读 hook，不替换数值路径  
> 性能声明：`performance_claimed=false`

## 1. 目标

在不修改 αβ-CROWN/auto_LiRPA 源码、不改变 provider 返回值与 optimizer 轨迹的前提下，证明 MR3
可以从真实 `activation_bab_bound / CROWN-optimized` 外层 exact call 精确绑定 P-anchor：

- 外层 provider call identity=`beta_split`，内部 backward evaluation=`10`；
- start node 固定为 `/49`；
- ReLU=`/input-24`，其输入/producer Conv=`/input-20`；
- compressed α=`[2,1,6,86]`，feature index 可无损恢复至 `[6,16,8,8]`；
- β 在 P-anchor 上为空，不能伪造 zero tensor；
- ReLU 输出 A 与随后 Conv 输入 A 的 identity/content/shape 邻接可证明；
- probe 只调用原方法并原样返回，provider result/state 与 control 等价。

## 2. 协议

固定两个独立 pair，顺序=`CP/PC`：`C` 为只记录外层 call 的 control，`P` 为安装 node-level
pass-through hook 的 probe。每侧从 isolated property copy 启动 fresh solver，固定 CUDA、seed=100、
`max_iterations=1`、batch=64、alpha steps=5、beta steps=10。

probe 必须记录每个 evaluation 的：start node、ReLU/Conv 名称与类、输入/输出 shape/dtype/device、
alpha/feature-index ABI、A pointer/version/content receipt、当前 CUDA device/stream。禁止记录 latency。

## 3. 门禁

1. 两个 pair 的 solver status、success、visited domains 与外层 exact-call result hash 等价；
2. probe 外层 beta exact call=`1`，内部 evaluation=`10`，P ReLU/Conv=`10/10`；
3. evaluation ordinal 连续为 0..9，全部 start node=`/49`；
4. ReLU output lower-A 与 Conv input lower-A 的 shape/pointer/version/content 全部一致；
5. α sparse/full ABI、bounds、weight、bias、lower-A、bias contribution 与冻结 MR3 合同一致；
6. P β 为空；fallback/eager/replacement/native-shadow=`0/0/0/0`；
7. probe 前后 CUDA device/stream 不漂移；
8. replay 从 raw 重算全部门禁，不能只验 digest；tamper 至少 10 类全重签仍被拒绝。

## 4. 机械结论

- 通过：`VALIDATED-MR3-0-PROVIDER-HOOK-FEASIBILITY`，开放 MR3 fail-closed candidate bridge
  implementation；
- hook 不可精确绑定：`BLOCKED-PRODUCTION-HOOK-MISSING`；
- probe 改变 provider 语义或无法证明邻接：`VALIDATED-NO-GO-MR3-0-HOOK-BOUNDARY`。

任何结论都不开放 timing、multi-site、S-anchor、same-solver performance 或 complete-query claim。

