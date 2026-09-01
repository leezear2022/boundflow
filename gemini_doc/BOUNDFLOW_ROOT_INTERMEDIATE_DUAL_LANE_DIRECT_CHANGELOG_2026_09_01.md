# BoundFlow `/input-28` 双通道 CROWN 直接执行变更记录

status: implemented-and-measured
date: 2026-09-01
external-audit-requested: false
performance-claimed: false

## 1. 本轮目标

在已经直接替换最终 root CROWN 的基础上，继续减少真实 α-CROWN optimizer 热路径中的原生
PyTorch/auto_LiRPA 遍历。本轮选择生产调用中的 `/input-28` 中间起点，将其 lower/upper 两条
CROWN 通道接入现有 residual、projection 和 input-domain TVM/TIR 执行器，并保持原 solver、
optimizer、branch、termination 与状态所有权不变。

本轮不是外审轮，也不升级论文性能 claim；它交付代码、真实 GPU 替换和三组 fresh 性能数据。

## 2. 生产语义盘点

真实 ResNet2B property 0 的一次 10-evaluation optimizer 轨迹包含五类重复中间起点：

| start node | spec count | 每次 optimizer 中的出现次数 |
|---|---:|---:|
| `/input-8` | 132 | 5 |
| `/39` | 121 | 5 |
| `/input-20` | 86 | 5 |
| `/44` | 178 | 5 |
| `/input-28` | 27 | 5 |

`/input-28` 路径跨越 15 个 bound graph 节点，同时计算 lower 与 upper；它复用五组真实 α：
`/45=178`、`/input-24=86`、`/input-16=121`、`/input-12=132`、`/input-4=164`。

## 3. 实现内容

1. 将 input-domain TIR 的 `spec_count` 从固定 3 泛化为任意正整数，同时保持 ResNet2B 的
   channel、空间、stride、padding、164 个 α 坐标和 CUDA ABI 继续 fail closed；新增 27-spec
   编译与 GPU forward/VJP 覆盖。
2. 新增双通道 compiled executor。lower 直接执行 lower primitive；upper 使用精确对偶恒等式
   `U(A, α_upper) = -L(-A, α_upper)`，通过翻转 α 的 lane 维把 native upper plane 映射到
   现有 lower primitive，并保留 autograd 到原 upper plane 的梯度映射。
3. 新增 production live bridge，限定只接管 exact `/input-28`、27 unstable spec、lower+upper 的
   OneHotC 调用；不满足合同即 fail closed。
4. 实现三种逐级模式：
   - `shadow`：原生结果返回，TVM 候选只比较；
   - `replace`：保留原生状态推进，但返回 TVM 候选并使用其 backward；
   - `direct`：完全不执行该调用的原生 bound traversal，TVM 结果直接发布给后续 optimizer。
5. three-fresh runner 新增 direct-intermediate 选项和 receipt 门禁，要求 5/5 调用命中、
   native execution=0、fallback=0。

## 4. 正确性证据

### 4.1 Shadow

- 5/5 production calls admitted；fallback 0；
- lower max abs diff `2.5033950805664062e-06`；
- upper max abs diff `1.7285346984863281e-06`；
- lower/upper sign exact。

### 4.2 Replace 与 direct

- replace：5/5，lower max diff `2.0266e-06`，upper max diff `2.2948e-06`；
- direct：native execution 0，5/5，fallback 0；
- direct 相对 shadow 的最终 solver lower max diff `7.15e-07`；
- decisions、queue、depth、status 一致。

### 4.3 三组 fresh 端到端

- 3/3 pair 的离散语义完全一致；
- 最终 lower 最大绝对误差 `1.7285346984863281e-06`；
- 最终 lower sign exact。

冻结 artifact 位于：
`artifacts/root-crown-intermediate-live/three-fresh-direct-v1/`。其 `summary.json` 文件 SHA256 为
`2b352f7749855dd01255cc53ac243aed3c4de608c45fe22e0fd2bb593906b09b`，内部 canonical
summary hash 为 `678fe300502301b89213958066d8c87f36982623e59e2a32bef3a71e10f2c398`。

## 5. 性能结果

对照为同一 αβ-CROWN solver 的 `B4-A-GC`；候选同时开启此前的 final-root direct 和本轮
`/input-28` direct：

| scope | 几何平均加速 | 最差 pair |
|---|---:|---:|
| complete query | `1.119954955878769x` | `1.0945451257531975x` |
| update-bounds core | `1.2108423107300137x` | `1.16987564821675x` |
| root measured region | `1.0959222558392934x` | `1.0854829482661197x` |

候选 complete-query 三次分别为 `523.790 ms`、`517.748 ms`、`512.780 ms`；对照分别为
`573.312 ms`、`582.106 ms`、`585.350 ms`。

相对上一阶段约 `1.10608x` 的 query geomean，本轮提高到约 `1.11995x`。但冻结的
`1.15x` complete-query 研究门槛仍未通过，因此 `performance_claimed=false`。

## 6. 内存与结构限制

- peak allocated ratio：`1.0260594861099397x`；
- peak reserved ratio：`1.0256410256410255x`；
- 即显存约增加 2.6%，没有 memory claim。

当前 dual-lane executor 是三个已有 custom VJP 的组合，每条通道仍跨越两处 dense-A 模块边界；
receipt 如实记录 `single_rematerializing_owner=false`。它证明了生产中间起点可以正确直接替换，
但还不是最终的“单一 structured owner + kernel 内重算 + minimal saved state”设计。

## 7. 下一步

下一性能刀固定为 `/44` sparse-Patches 起点：

1. 捕获并冻结 `/44` 的 Patches/identity 语义和 178-spec 真实状态；
2. 以稀疏 Patches 直接进入 residual compiled region，禁止先在 Python 中长期展开 dense A；
3. 重复 `shadow → replace → direct`，先证明 lower/upper/VJP/trajectory，再计时；
4. 与本轮 `/input-28` direct 组合跑三组 fresh。

当前 `/44` 原生事务约 `24.264 ms`，是最可能把 complete query 从 `1.120x` 推过 `1.15x`
的单一剩余区域。若它不能传播收益，再转向将本轮三段 custom VJP 合并为单一 rematerializing
owner，减少 dense-A crossing、launch 和保存状态；不以外审替代优化。
