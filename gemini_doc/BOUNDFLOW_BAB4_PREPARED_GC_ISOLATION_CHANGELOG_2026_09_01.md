# BAB4 prepared-runtime GC isolation 修改记录

status: diagnostic-qualified-formal-open
date: 2026-09-01
external-audit-requested: false
performance-claimed: false

## 1. 证据与目标

三对 complete-prelude attribution 显示，BAB4 候选在 αβ-CROWN 原生 `gc.collect()` 中稳定多花约
`10.4--11.6 ms`。原因是 query 前已经常驻的 TVM modules、typed plans、cache 和 runtime wrapper
扩大了 Python 全代 GC 的扫描对象图；query 内新产生对象的回收语义本身不能删除。

目标是隔离“确定长期存活的 prepared 对象扫描”与“query 内新对象回收”，而不是跳过原生 GC。

## 2. 实现

初版曾尝试可撤销 `gc.freeze()/unfreeze()`，但测试进程和 Torch runtime 可能已有其他 owner 的 frozen
objects，而 Python 不能只解冻本模块新增的子集。该方案在实现测试阶段撤销，未进入 GPU 诊断。

最终 `prepared_gc_isolation_v1()` 使用 generation isolation：

1. query 计时前支付完整 `gc.collect()`；
2. 只把 `complete_verifier_func.gc` 模块引用替换为局部 proxy，不修改全局 `gc.collect`；
3. αβ-CROWN 原位置的无参 `gc.collect()` 被收窄为 `gc.collect(1)`，收集 query 后新产生的 gen0+gen1；
4. 重复调用、带参数调用均 fail closed；
5. query 结束恢复原模块对象，再在计时外做完整 GC；
6. fail closed 验证模块身份与 GC enable state 恢复；
7. receipt 披露准备/query/恢复三次 collection 的耗时、generation 与收集数量。

新增公平配置 `B4-A-GC` 与 `BAB4-GC`：两边都使用 prepared request、原生 root warmup 与相同 GC
isolation；候选另外使用既有四段 BAB4 TIR。先运行三对带 complete-prelude attribution 的诊断，只有
语义、GC 恢复、显存和 query 都不恶化时才进入五对正式计时。

该设计不会接管或解冻其他 owner 的 frozen set，生命周期测试直接验证局部 module proxy、query-local
young-generation collection、完整恢复和计时外最终 full collection。

## 3. 三对诊断结果与正式门禁

三对交替 fresh 诊断位于 `/tmp/bab4-gc-prelude.Ktu06H/profile`，它不是正式 headline artifact，
但已通过进入五对正式计时的资格：

- complete-query 几何平均 `1.062499x`，三对全部快于 control；
- core 几何平均 `1.193498x`；
- 原 `gc.collect()` 候选额外约 `11 ms` 的差距降为中位 `0.004148 ms`；
- lower 最大误差 `1.4901161193847656e-06`，符号与离散语义全部一致；
- peak allocated/reserved 几何平均比值分别为 `1.005078x/1.010256x`；
- 两边 receipt 都证明 generation=1、query call=1、owner 已恢复；
- query 外完整 prepare/restore GC 仍执行并披露，不混入 query headline。

因此开放 `B4-A-GC` 对 `BAB4-GC` 的五对正式 artifact。正式生成器会把 GC receipt 作为 replay
语义的一部分 fail closed；该阶段仍不把 `1.062499x` 诊断数升级为最终性能 claim。
