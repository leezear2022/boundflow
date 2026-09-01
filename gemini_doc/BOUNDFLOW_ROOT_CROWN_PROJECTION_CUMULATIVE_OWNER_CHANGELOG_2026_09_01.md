# BoundFlow root CROWN projection residual 与累计 owner 变更记录

status: implemented-and-locally-validated
date: 2026-09-01
external-audit: not-requested
performance-claimed: false

## 1. 本轮解决的问题

上一提交只接管了 root CROWN 尾部：terminal Linear/ReLU 与 `/45 → /input-16` 的普通
residual block。更早的 `/input-16 → /input-4` 仍由 PyTorch/auto_LiRPA 执行，其中既有
stride-2 主分支，也有 1×1 stride-2 projection skip 分支。

本轮完成了该 projection residual 的生产捕获、独立数学闭合、TVM/TIR forward/full-VJP、
prepared runtime、same-solver 接入，以及 terminal + 普通 residual + projection residual 的单一
custom-autograd owner。

## 2. 真实生产事务

固定模型/property 下的 root start node 为 `/49`，被接管的新增拓扑是：

```text
/input-16 ReLU
  → /39 Add
      ├─ /37 Conv 3×3, stride 1
      │    → /input-12 ReLU
      │         → /input-8 Conv 3×3, stride 2
      └─ /38 Conv 1×1, stride 2
  → /input-4 boundary
```

捕获结果：

- 一个真实 optimizer transaction；
- 5 次 forward evaluation；
- 4 次 optimizer backward；
- 七个选中节点各命中 5 次；
- incoming coefficient：`[3,1,16,8,8]`；
- output coefficient：`[3,1,8,16,16]`；
- entry α：`[2,3,1,121]`；
- inner α：`[2,3,1,132]`；
- outer/main/skip 权重分别为 `[16,16,3,3]`、`[16,8,3,3]`、`[16,8,1,1]`。

## 3. 独立数学闭合

`scripts/probe_root_crown_projection_oracle.py` 没有调用新 TIR，也不复用新 backend 的计算式。
它使用 PyTorch 的 ReLU bound 闭合表达式和三次 `conv_transpose2d` 重建：

1. entry ReLU slope/intercept；
2. 3×3 stride-1 outer Conv；
3. inner ReLU slope/intercept；
4. 3×3 stride-2 main Conv；
5. 1×1 stride-2 skip Conv；
6. 五部分 bias delta；
7. incoming、entry/inner α 与 bound 的 VJP。

对生产 capture 的局部所有权字段：

- forward 最大误差：`2.384185791015625e-7`；
- incoming 与 α 梯度符号全一致；
- capture 中 lower/upper 的总梯度包含区域外消费者，因此只披露、不拿来要求局部 oracle 等于总梯度。

## 4. TVM/TIR 实现

新增 `RootCrownProjectionTemplateV1`，其静态合同显式区分：

- output-side geometry：16×8×8；
- input-side geometry：8×16×16；
- main/skip kernel、padding 与 stride；
- entry/inner sparse α coordinates；
- spec/domain 两条独立轴；
- compute capability 与 schedule thread extent。

forward TIR 的 projection skip 不作为长期 Python tensor 保存。主分支和 skip 分支在同一编译模块内
产生局部结果并合并为 boundary coefficient；full VJP 在反向中把 output adjoint 分别投影回 main 与 skip，
再在 entry transformed adjoint 汇合。

prepared runtime 保留 DLPack zero-copy、current-stream 双向核对、persistent output/VJP arena、
cache 和 fail-closed shape/dtype/device/contiguity 检查。

## 5. 局部 correctness 与性能

`artifacts/root-crown-projection-capture/resnet2b-prop0-v1/tir-probe.json`：

- 5/5 evaluation；
- forward、incoming、entry/inner α 与局部 bound VJP 对独立 oracle 全部 sign exact；
- 最大 oracle 绝对误差：`2.1457672119140625e-6`；
- fallback：0；
- DLPack pointer：2614/2614 exact；
- candidate median：`0.56012797 ms`；
- native PyTorch oracle median：`1.27027202 ms`；
- 局部 forward + full-VJP speedup：`2.26782463x`。

该局部数字不等于完整查询性能，`performance_claimed=false` 保持。

## 6. same-solver 单一累计 owner

先运行了 two-owner 诊断：旧 terminal+residual owner 和新增 projection owner 各形成一个 autograd node。
它的完整查询 geomean 只有 `1.057734x`，说明第二个 wrapper 吃掉了新增局部收益。

随后新增 `RootCrownExpandedSuffixTIRExecutorV1`：

```text
terminal TIR
  → residual TIR
  → projection TIR
  → one custom-autograd owner
  → projection VJP → residual VJP → terminal VJP
```

它仍是三个 prepared TVM module，而不是虚假宣称为一个 CUDA kernel；改进的是 autograd ownership、
中间 coefficient 的保存边界和 backward 链接。`/38` 在 host graph 中只作为已编译 boundary coefficient
的零计算 carrier，原 1×1 Conv 已在 projection TIR 内执行，生产路径不会重复计算。

## 7. 三组 fresh same-solver 结果

冻结顺序：`control→candidate / candidate→control / control→candidate`。每个子进程重新初始化 solver，
compile/prepare 不计入 query headline。

结果来自 `artifacts/root-crown-expanded-live/three-fresh-single-v1/summary.json`：

| scope | geomean speedup | worst pair |
|---|---:|---:|
| complete query | `1.066130x` | `1.055696x` |
| root incomplete | `1.100645x` | `1.092305x` |
| optimized-bounds transaction | `1.167842x` | `1.152715x` |
| autograd backward | `1.280538x` | `1.273173x` |

语义：

- 三对 discrete semantics 全一致；
- final lower 最大绝对差：`1.6689300537109375e-6`；
- 每个 candidate 为 5 forward / 4 backward；
- cumulative autograd owner：每次 evaluation 恰为 1；
- fallback：0。

## 8. 与上一提交的关系

上一提交的完整 query geomean 为 `1.057805x`。本轮达到 `1.066130x`，增加约 0.8 个百分点；
因此 projection 局部 `2.2678x` 的收益只有一小部分传播到 complete query。原因不是 kernel 错误，而是：

- 仍有 `/input-4 → /input` 和输入域收尾未编译；
- 三个 prepared module 仍分别 launch，尚未做跨 module TIR 合并；
- root 之外的 model build、BaB、branch/queue 与环境固定成本不变；
- 当前 workload 很小，wrapper/launch 占比高。

## 9. 当前结论与下一步

成立：

- stride-2 projection residual 可以被 verification-aware TIR 正确表示；
- main + skip 的编译事务和 full VJP 成立；
- 单一 autograd owner 优于两个串联 owner；
- complete query 从约 `1.058x` 提升到 `1.066x`。

不成立：

- 还没有 10× complete-query；
- 还不是一枚跨三块的 CUDA kernel；
- 还没有证明跨 workload 泛化；
- 还不能升级 ASPLOS 性能 claim。

唯一工程下一步是捕获并评估最早的 `/input-4 → /input` Conv/input-domain transaction；若该区域的
same-solver share 与可达 speedup 数学上不够，则停止继续按层铺 TIR，转向跨 module launch fusion、
CUDA Graph 或更高 share 的 BaB/runtime 事务。

## 10. 工程验证

- projection/旧 residual/累计 suffix targeted：`46 passed`；
- 全量：`2170 passed, 4 skipped`；
- 4 个 skip 均为既有环境边界：TVM 重复编译规避、两项冻结 VNN-COMP checkout 不可用、
  `BOUNDFLOW_CUDNN_ROOT` 未配置；
- Black：clean；
- mypy（12 个本轮文件）：clean；
- Pylint：`10.00/10`；
- `git diff --check`：PASS。

## 11. 证据哈希

- capture.pt：`f37fae95bb2109668ae0a4a75abb9ac71e0dd9374392199ecd56d259057991d3`
- receipt.json：`e1d002936e7f3761c517ee99e5d357d569d7587272a1a0f97159c38c764a56d5`
- oracle.json：`17d147e9b3b0771488665e4399d399e3eda413d0429b069f9e5a2545725c6ed8`
- tir-probe.json：`41579d6f958bf3668884513f51a834a3e8f2c1d6e61fbef20ad094a7b074a7db`
- three-fresh summary：`d18dccab0cac895ea3bd1c24e8bce9cf8cb1a04106b9ef2e5bbd281149e765b6`
