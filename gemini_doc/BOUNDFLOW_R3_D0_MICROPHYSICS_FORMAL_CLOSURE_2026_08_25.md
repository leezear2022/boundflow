---
status: validated-r3-d0-compiled-region-schedule-opportunity
updated: 2026-08-25T15:10:00+08:00
type: closure
topic: boundflow
slug: r3-d0-microphysics-formal-closure
stage: s01
---

# R3-D0 Microphysics 正式关闭

## 1. 判定

R3-D0以`VALIDATED-R3-D0-COMPILED-REGION-SCHEDULE-OPPORTUNITY`关闭。它只开放新的
`R3-D1 compiled-region schedule/fusion`候选，不恢复R3-2B claim，不开放R3-3、multi-site或
same-solver计时。

formal source=`423206911599de5aee7a718ddee65cf527b48e28`，artifact：
`artifacts/r3-structured-owner/r3-d0-microphysics-v1`，summary hash=
`67f433d708fe544951d07425c8cb7644717072417d6e68287e5d2e82693c1e9b`。

## 2. Formal 结果

固定cooldown后的5 fresh pair全部满足R3-2B reference `±15%` sanity和时钟/归属门禁：

| pair | native ms | candidate ms | compiled share | required region speedup |
|---:|---:|---:|---:|---:|
| 0 | 100.159 | 739.882 | 0.997180 | 9.0661x |
| 1 | 100.434 | 735.977 | 0.997216 | 8.9891x |
| 2 | 100.176 | 752.219 | 0.997257 | 9.2138x |
| 3 | 100.090 | 736.293 | 0.996627 | 9.0678x |
| 4 | 97.072 | 734.798 | 0.996897 | **9.3180x** |

- Graph/dispatch physical gate：5/5 false；candidate projected host residual约`1.92–2.36 ms`，无限消除
  也达不到目标；
- compiled region：profiled wrapper share=`99.6627%–99.7257%`，worst required=`9.3180x ≤ 10x`；
- candidate fallback=`7/606`量级、native fallback固定`400/8516`量级，均低于5%，unattributed=0；
- `residual6`占`66.06%–67.02%`，`residual11`占`28.27%–29.10%`，
  `effective_pre23`占`3.41%–3.51%`；
- residual6+residual11若单独承担全部目标，worst required=`15.4733x`；
- 12/12 fully re-signed tamper拒绝；targeted=`9 passed`；`performance_claimed=false`。

## 3. 失败与修正轨迹

1. 首批formal暴露profile/unprofiled scope混用，旧replay出现share>1；修正为同scope share后整批重跑；
2. 首轮tamper中correlation-id攻击被接受；将完整canonical event hash绑定进ledger后从新source重跑；
3. 随后批次pair 3/4 native发生热漂移，被`±15%`门禁整批拒绝；冻结每worker固定30秒cooldown后从pair 0
   重跑；
4. 最终批次和12类tamper全部通过。

无失败批次进入claim，也没有裁剪pair、放宽阈值或复用失败raw。

## 4. 物理解释

当前R3 candidate慢的主因不是Python/JIT/Graph dispatch，而是两个手写residual PrimFunc：它们把两层
卷积转置与中间ReLU slope展开成单线程内嵌套串行循环，并在每个输出位置重复计算中间系数。这个实现满足
语义和“不跨层持久化dense A”，但不是经过GPU schedule优化的算法。

D0证明的是“whole compiled recurrence存在数学可达的schedule opportunity”，不是已经获得性能，也不是
CUDA Graph机会。

## 5. 唯一下一动作

执行`BOUNDFLOW_R3_D1_COMPILED_REGION_SCHEDULE_PLAN_2026_08_25.md`：先把residual11重写为两阶段
scratch-bounded factorization并做correctness，再扩residual6和whole-region累计timing。D1-C通过前
`performance_claimed=false`、R3-3和same-solver保持关闭。
