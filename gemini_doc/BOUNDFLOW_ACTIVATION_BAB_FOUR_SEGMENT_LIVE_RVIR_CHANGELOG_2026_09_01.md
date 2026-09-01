# Activation-BaB 四段 TIR 接入 live RVIR exact-call 修改记录

status: implementation-validated-dev-performance-only
date: 2026-09-01
performance-claimed: false
external-audit-requested: false

## 1. 本轮解决的问题

上一阶段已证明 terminal、residual、projection、input-domain 四个 activation-BaB
segment 可以在单一 custom-backward owner 下执行，并在局部 warm 诊断中获得约 `1.37x`。
但该数字不经过真实 host solver、RVIR state transaction、B4-A handoff、KFSB 与 query
计时，不能回答收益是否能传播到求解器。

本轮把四段 owner 接到现有 production seam：

```text
alpha-beta-CROWN stage_solve.update_bounds_core
  -> RVIR pre-state / typed plan / persistent region
  -> BAB4 four-segment TIR optimizer (10 evaluate / 9 Adam mutation)
  -> terminal lower + six native lA
  -> existing B4-A handoff / lineage / atomic commit
  -> unchanged branch / termination / queue owner
```

没有修改 αβ-CROWN、auto_LiRPA、TVM vendored source，也没有把 BaB host 控制流迁进
BoundFlow。

## 2. 代码改动

### 2.1 四段 production optimizer

新增 `boundflow/runtime/bab_four_segment_optimizer.py`：

- 从既有 `R31FullRegionPlanV1` 的 generic tensor/layout metadata 构造四个真实 TIR
  executor；
- 绑定 production ResNet2B 六个 ReLU site `17/19/23/25/28/31`；
- 保持六组 raw α 与一组 active β 为唯一 mutable owners；
- 执行冻结的 10 次 lower evaluation、9 次 Adam mutation、指数学习率衰减与 clamp；
- 最终输出 plan-order compact α/β、lower 以及 KFSB 所需六份 native-layout `lA`；
- receipt 如实记录四段、76 次 forward launch、36 次 backward launch、fallback 0；
- 四个 template hash 与 scheduled-TIR hash 合并成 compiled identity。

### 2.2 live exact-call bridge

新增 `boundflow/runtime/bab_four_segment_exact_call_bridge.py`：

- 复用现有 S4 AOT/persistent physical plan 和 prevalidated core instance；
- 在 exact call 内只做 dynamic tensor rebind、10/9 optimizer 和 handoff；
- compile 与静态 module preparation 保持在 query 计时之外；
- 继续使用既有 `_dense_terminal_state`、forward trace、lineage、B4-A validation 和
  atomic live/host commit；
- 新 receipt 同时绑定 base S4 assets 与四段 TIR assets，且
  `performance_claimed=false`。

### 2.3 terminal lA 导出

扩展 `BabFullRegionTraceV1`，直接从四段 forward intermediates 导出：

| native site | 四段 owner 内部值 |
|---|---|
| 17 | projection 输出 coefficient |
| 19 | projection outer-conv coefficient |
| 23 | residual 输出 coefficient |
| 25 | residual main-conv coefficient |
| 28 | terminal 输出 reshape 后的 residual incoming |
| 31 | objective 经最后 Linear 后的 terminal incoming |

这些值本来就是当前 forward 的中间结果，不新增一轮 CROWN，也不新增 kernel。
导出时强制 native contiguous memory format，stride 与原生 handoff 一致。

### 2.4 same-solver worker

`scripts/run_asplos27_s4_same_solver_worker.py` 新增开发配置 `BAB4`。静态阶段用
不依赖 source capture 的合法 dummy state 预热 module loading、DLPack views 和 persistent
arenas；模型 parameter 不被改写。真实 query 到来后全部 dynamic state 会被 production
snapshot/live source 覆盖。

## 3. 正确性结果

### 3.1 exact-call 专项

真实 production fixture 中，新 bridge 与原生 αβ-CROWN/B4-A 对比通过：

- terminal lower：`allclose(atol=2e-4, rtol=2e-4)`；
- lower sign：exact；
- 六个 native ReLU site 的 `lA`：逐项 allclose；
- 六个 `lA` sign：exact；
- 六个 `lA` stride：与原生 handoff exact；
- compiled forward/backward：`76/36`；
- fallback：`0`；
- targeted suite：`63 passed`。

### 3.2 三组 fresh same-solver 开发配对

配置为原始 `B4-A` control 与 `BAB4` candidate，独立新进程、顺序交替。lower 最大差：

| pair | lower max abs diff | sign exact | control env | candidate env |
|---:|---:|:---:|:---:|:---:|
| 0 | `1.13249e-6` | yes | admitted | admitted |
| 1 | `1.22190e-6` | yes | admitted | admitted |
| 2 | `2.08616e-6` | yes | admitted | rejected (thermal/power signal) |

第三对的数值语义通过，但性能样本不得进入 formal headline。

## 4. 开发性能诊断

下表完整披露三对原始 wall time；单位为 ns：

| pair | B4-A core | BAB4 core | core speedup | B4-A query | BAB4 query | query speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 245254665 | 215648866 | `1.13729x` | 1418768686 | 1116885996 | `1.27029x` |
| 1 | 249344262 | 217741231 | `1.14514x` | 1426016283 | 1120375343 | `1.27280x` |
| 2 | 251130924 | 216102211 | `1.16209x` | 1440144941 | 1119536624 | `1.28638x` |

未经环境筛选的开发统计：

- core geomean：`1.14813x`，worst pair：`1.13729x`；
- complete-query geomean：`1.27647x`，worst pair：`1.27029x`。

这组数字说明收益已经从四个 kernel segment 传播到真实 host solver，并在本地开发测量中
越过 `1.15x complete-query` 研究线；但由于只有 3 pairs、其中 1 个 candidate 环境未准入、
也没有冻结 raw/replay/tamper，本轮**不升级性能 claim**。

## 5. 首跑慢与预热修正

未做四段 runtime 预热时，首个 live core 为约 `750 ms`；增加合法 dummy static warmup 后，
同一路径降到约 `212--218 ms`。这说明主要差异来自首次 module launch、DLPack view 与 arena
初始化，而不是 production lower transaction 本身。该 warmup：

- 位于 query 计时之前；
- 不读 source capture；
- 不改模型 parameter；
- receipt 明确披露 warmup 时间和 `source_capture_runtime_dependency=false`。

## 6. 当前边界

1. 这不是 formal artifact，不能写入论文 headline；
2. 当前只覆盖 ResNet2B、domain=6/spec=1 的已验证 topology；
3. 四段模块仍是 segment-level 编译，尚未合成一个跨 segment mega-kernel；
4. optimizer 仍由 PyTorch Adam host orchestration 驱动；
5. query speedup 中包含初始/incomplete verifier 的运行波动，因此必须用冻结配对协议复核；
6. 没有证明 queue geomean、held-out family、solved/TTV 或 10x 总体目标。

## 7. 下一步

唯一直接后继是生成 `B4-A vs BAB4` 的 five-fresh formal candidate：

1. 冻结 source revision、四段 compiled identity、model/property 与 pair order；
2. 至少 5 个独立 control/candidate pairs，环境不准入时重试而不是纳入 headline；
3. raw-first 保存 lower、discrete semantics、core/query wall、warmup receipt 与四段 launch
   inventory；
4. stdlib replay 独立重算 correctness、geomean、worst pair 与环境 admission；
5. coherent tamper 覆盖 raw timing、lower、compiled identity、launch count 与
   `performance_claimed`；
6. 若 query geomean `>=1.15x` 且 worst pair `>=1.00x`，再开放下一层跨 segment
   transaction fusion / optimizer orchestration 优化；否则先 profile BAB4 内部热路径。

本轮按用户要求不发起外审；待下一轮 formal candidate 完成后再统一交审。
