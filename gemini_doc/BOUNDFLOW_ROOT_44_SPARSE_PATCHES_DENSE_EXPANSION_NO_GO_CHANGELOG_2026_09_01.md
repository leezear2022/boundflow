# BoundFlow `/44` Sparse-Patches 接管与 Dense-Expansion NO-GO 记录

status: implemented-correctness-passed-performance-no-go
date: 2026-09-01
external-audit-requested: false
performance-claimed: false

## 1. 本轮问题

上一阶段把 `/input-28` lower/upper CROWN 接入 compiled path 后，完整 query 的三组 fresh
geomean 为 `1.1199549559x`，但尚未达到 `1.15x`。生产归因显示 `/44` 是剩余 intermediate
prior-bound 热点之一，因此本轮尝试直接替换 `/44`。

真实捕获证明 `/44` 不是 dense coefficient 起点，而是 auto_LiRPA `Patches`：

- 前四次 optimizer evaluation：`[178, 1, 16, 1, 1]`；
- 最后一次：`[175, 1, 16, 1, 1]`；
- 每个 spec 带 `(channel, height, width)` 稀疏位置；
- lower/upper 同时计算；
- 路径依次经过 residual、projection residual、input-domain concretization。

## 2. 实现

本轮实现了一个严格的、opt-in 的 correctness probe：

1. 新增 `RootCrownSparsePatchesSeedTemplateV1`，只准入 16×8×8、domain=1、1×1 identity
   Patches；
2. 新增 TVM/TIR seed kernel，把 `Patches + (h,w)` 写入持久 dense seed arena；
3. 对 178 和 175 两个真实 shape 做计时外 specialization/compile/prepare；
4. 复用现有 residual/projection/input-domain dual-lane TIR 与 custom backward；
5. `/44` 后没有 `/45` ReLU，因此在既有 residual primitive 的第一层使用 stable-zero identity
   合同，保证 slope=1、intercept=0、dummy α 不拥有梯度；
6. 新增 `shadow → replace → direct` 三种生产桥接模式和 receipt；
7. direct 模式发布 `/input-24`、`/input-16`、`/input-12`、`/input-4` 的 optimizer-visible
   lA/uA 状态，并把未遍历节点清空。

该实现没有修改 auto_LiRPA、αβ-CROWN 或 vendored TVM。

## 3. 正确性结果

### 3.1 Shadow

- 5/5 调用命中：178-spec×4、175-spec×1；
- fallback=0；
- lower max abs diff `1.430511474609375e-06`；
- upper max abs diff `1.5497207641601562e-06`；
- lower/upper sign exact。

### 3.2 Replace

- native 仍执行以推进状态，但候选 forward/backward 返回给 optimizer；
- 5/5，fallback=0；
- 相对 native lower max diff `1.1920928955078125e-06`；
- upper max diff `1.3113021850585938e-06`；
- 最终 query lower 相对 shadow 最大差 `7.152557373046875e-07`；
- branch decision 一致。

### 3.3 Direct

- native `/44` execution=0；
- 5/5，fallback=0；
- 最终 query lower 相对 shadow 最大差 `1.3113021850585938e-06`；
- branch、queue、depth、status、success 全一致。

因此 `/44` 的 Patches 起点、四组真实 α、两条 polarity、custom backward 和 optimizer state
接管在当前 workload 上成立。

## 4. 三组 fresh 性能

候选同时开启：final-root direct、`/input-28` direct、本轮 `/44` direct。对照仍为同一 solver
的 `B4-A-GC`。

| scope | geomean speedup | worst pair |
|---|---:|---:|
| complete query | `1.0954451085x` | `1.0756790015x` |
| update-bounds core | `1.1627763717x` | `1.1226452599x` |
| root region | `1.0362454768x` | `1.0250367317x` |

三组 candidate query 为 `540.856 / 522.960 / 526.915 ms`。当前 candidate query 几何平均
`530.188 ms`；上一阶段 winner 为 `518.087 ms`。也就是说加入本轮 `/44` 路径后，candidate
自身慢了 `1.02336x`（约 2.34%）。query `1.15x` gate 未通过。

冻结 summary：
`artifacts/root-crown-sparse-patches-live/three-fresh-dense-expansion-v1/summary.json`，文件
SHA256 为 `2eca8023ef0976a8df337b08ddcb3546601e3333bdcd907524d4f0553a2ef4f3`，canonical
summary hash：
`530aa82deda17e6175bcb87f4311a7aabe3e61f7725d5355ac0d9ad594ca7eca`。

## 5. 内存结果

- peak allocated ratio：`0.4981160432x`；
- peak reserved ratio：`1.0666666667x`。

跳过 native Patches autograd/materialization 使动态 allocated peak 约减半，这是实质性正信号；但为
178/175 两套 specialization 同时保留 compiled modules/arenas，使 reserved peak 增加约 6.7%。所以
本轮不能宣称完整 memory gate 通过。

## 6. NO-GO 原因

失败不在 sparse seed kernel 的正确性，而在 seed 后立即恢复 dense 表示：

```text
native Patches footprint:
1×1 → 3×3 → 5×5 → (7×7 + 9×9) → 15×15 → 31×31

本轮 probe:
1×1 sparse → dense 16×8×8
            → dense residual
            → dense 8×16×16
            → dense input coefficient
```

原生 Patches 在前半段只计算局部 footprint；本轮为了复用已有 dense CROWN TIR，把大量确定为零的
位置重新带入卷积、VJP 和模块边界。每次 `/44` 还组合两条 polarity、三个 compiled owner；五次累计
新增的 dense arithmetic、launch 与 crossing 吞掉了省下的 Python/Patches traversal。

因此本轮路线正式判定：

> `sparse Patches → dense seed → existing dense TIR chain` 为性能 NO-GO；不得作为默认 candidate，
> 也不得用正确性通过升级性能 claim。

代码保留为 opt-in correctness oracle、稀疏 admission/seed ABI 和后续 sparse TIR 的对照，不进入默认
生产路径。

## 7. 下一路线：Patch-Footprint-Preserving TIR

下一刀不再接 dense executor，而是把 Patches 表示保持到 input concretization：

```text
SparsePatchIR(spec location + patch tensor + stride/padding)
  → residual sparse TIR:       1×1 → 5×5
  → projection sparse TIR:     5×5 → 15×15
  → input sparse TIR:          15×15 → 31×31 → on-the-fly concretize
  → scalar lower/upper + compressed dα
```

设计约束：

1. 外部不得出现 `[spec,1,16,8,8]` 或 `[spec,1,8,16,16]` dense coefficient；
2. 每阶段 workspace 必须与实际 patch footprint 一致；
3. skip/add 通过 padding-aligned local patch merge 完成；
4. ReLU slope/intercept 根据 spec 的起点位置、stride、padding 在 kernel 内索引，不先 unfold 全图；
5. input-domain 不生成 32×32 dense A，31×31 patch 值生成后立即消费到 center/radius reduction；
6. custom backward 重算局部 slope/patch，不跨阶段保存完整 dense A；
7. 先做 178/175 forward/VJP 对 native Patches oracle，再进入 production shadow；
8. 单 `/44` wrapper-inclusive 必须快于 native；否则不开放 direct；
9. 只有与上一阶段 `1.1199549559x` winner 组合后更快，才进入新的三组 formal。

## 8. Claim 边界

- `performance_claimed=false`；
- 本轮证明的是 `/44` production semantics 可被编译接管，以及 dense expansion 的物理边界；
- headline 仍是上一阶段 query `1.1199549559x`，不是本轮 `1.0954451085x`；
- 不请求外审，先完成真正的 sparse patch-footprint kernels。
