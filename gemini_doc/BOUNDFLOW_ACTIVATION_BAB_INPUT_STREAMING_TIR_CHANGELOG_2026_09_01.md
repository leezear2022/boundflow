# BoundFlow activation-BaB input streaming TIR 变更记录

status: implemented-and-locally-validated
date: 2026-09-01
stage: s05
performance-scope: full-region-local-diagnostic
complete-query-performance-claimed: false
external-audit-requested: false

## 1. 本轮解决的问题

上一提交已经把真实 activation-BaB lower region 的 terminal、residual、projection 三段接到 TIR，
但最后的 input Conv + L∞ concretization 仍由 PyTorch 执行。因此，当时的 `1.391908x` 只能说明
terminal 小段变快，不能回答完整 lower region 是否变快。

本轮新增第四段 input streaming TIR，并把四段同时接到一个
`PreparedBabFullRegionOwnerV1` custom-backward 边界。现在该 owner 的完整路径是：

```text
terminal ReLU + beta + Linear TIR
  → residual TIR
  → projection TIR
  → input ReLU + ConvTranspose coefficient generation
      + L∞ concretization streaming TIR
  → compact final lower
```

这仍是 capture 驱动的真实 production-shape region，不是 live αβ-CROWN complete query。

## 2. Input TIR 的设计

新增：

- `boundflow/backends/tvm/bab_input_domain.py`
- `boundflow/runtime/bab_input_domain_tir.py`
- `tests/test_bab_input_domain_tir.py`

ABI 固定为真实 capture 的通用 shape/signature，而不是模型或 node id 特判：

- spec = 1；
- domain = 6；
- incoming coefficient = `[1, 6, 8, 16, 16]`；
- input image = `[6, 3, 32, 32]`；
- compressed α = `[2, 1, 6, 164]`；
- Conv weight = `[8, 3, 3, 3]`。

forward 不生成 `[1, 6, 3, 32, 32]` dense input coefficient tensor。每个输入位置的系数在
register-local scalar 中生成后，立即消费到：

```text
coefficient * center - abs(coefficient) * radius
```

随后只保留 block-local reduction。backward 同样重新计算局部 coefficient，直接产生
incoming VJP 和 compressed-α VJP；不把 dense A 保存到 outer autograd boundary。

本轮没有修改历史 `root_crown_input_domain.py`，因为其源码 identity 已被既有 formal artifact 绑定。
新实现复用其已验证语义，但使用独立 symbol、schema 和 hash。

## 3. 编译身份与 workspace

本机 RTX 4060 Laptop GPU / `sm_89` 编译结果：

- template hash: `8131ddd6a5204f068f2612d488e6c3faa58979d0e8f1dae76519e102fabc1d2a`
- unscheduled TIR hash: `810b5df9149681faeeab88acaf37aa4110c348a43c2ce44e1df951005205aae5`
- scheduled TIR hash: `b0453bcfd83036523db0e7d786639cbb38545d7d287cac2d64c8703f5ce94245`
- device source hash: `68a567ca455cf636b16e7181322f05153379c997555ca66d624fb79ab5e44815`

scheduled workspace inventory 只有 local/shared scratch：

```text
adjoint       [1]
bias_sum      [1]
coefficient   [1]
concrete_sum  [1]
partial       [2, 128]
reduction     [2]
```

没有 global dense input coefficient workspace。

## 4. 真实 capture 正确性

数据源：

`artifacts/bab-full-region-capture/resnet2b-prop0-v1/capture.pt`

### 4.1 Input 段

在 10 次真实 forward 和 9 次真实 VJP 上独立比较：

- concrete lower；
- output bias；
- incoming coefficient gradient；
- compressed α gradient。

结果：

- 最大绝对误差：`9.5367431640625e-07`；
- 全部 lower/gradient sign exact；
- launch：forward `10`、backward `9`；
- DLPack pointer exact：`227/227`；
- fallback：`0`。

### 4.2 四段完整 owner

四个 compiled executor 同时接入一个 custom-backward owner 后：

- final lower 与纯 PyTorch owner 在冻结容差内一致；
- 8 个动态 owner 的 gradient 全部在冻结容差内一致；
- 单次 outer forward+backward 的 compiled launch 为 forward `8`、backward `4`；
- `compiled_segment_count=4`；
- `saved_dense_coefficient_count=0`；
- fallback 为 `0`。

9 步真实 Adam mutation replay：

- 最大累计状态漂移：`3.259629011154175e-06`；
- 六组 compressed α 与一组 sparse β 的符号全部一致；
- compiled launch 累计 forward `72`、backward `36`。

## 5. 性能诊断

### 5.1 比较边界

对照和候选使用：

- 同一真实 capture ordinal 0；
- 同一个 `PreparedBabFullRegionOwnerV1` outer custom-backward 边界；
- 同一组 dynamic/frozen tensors；
- 同时执行 forward + `autograd.grad(-lower.sum())`；
- 5 次 warmup + 30 次逐次同步 wall-clock 样本；
- compile/prepare 不进入 warm timing。

唯一差异：

- control：四段均走 PyTorch reference；
- candidate：四段均走 TVM/TIR executor。

### 5.2 结果

| 指标 | PyTorch owner | 四段 TIR owner | 加速 |
|---|---:|---:|---:|
| warm median | `5.980895 ms` | `4.345010 ms` | `1.376497x` |
| warm geomean | `5.944319 ms` | `4.326956 ms` | `1.373788x` |
| 30-run range | `5.612834–6.121473 ms` | `4.129072–4.665524 ms` | 披露 |

单次 warm allocator 诊断：

- control peak allocated above baseline：`1,335,808 B`；
- candidate peak allocated above baseline：`4,608 B`；
- control end allocated delta：`0 B`；
- candidate end allocated delta：`512 B`。

这些是本机、单 capture、region scope 的开发期诊断。它们证明“第四段接通后完整 region 已有正收益，
且 dense coefficient lifetime 被消除”，但不升级 complete-query、queue、10x 或论文性能 claim。

## 6. 验证

- 新 input/four-segment 专项：`7 passed`；
- BaB + root-CROWN targeted：`70 passed`；
- mypy（4 个 touched source/test）：clean；
- pylint：`10.00/10`；
- black：clean；
- `git diff --check`：PASS；
- 全量 pytest：`2205 passed, 3 skipped, 6 warnings`；
- DocOps lint：提交前执行。

## 7. 下一步

下一步不再让用户为每个小段外审。先完成一个更有研究含量的批次：

1. 把四段 prepared owner 接到 RVIR/αβ-CROWN live exact-call bridge；
2. 冻结 control / historical B3 / four-segment candidate 三方 same-solver 协议；
3. 实测 complete-query share、integration overhead 和 B0 parity；
4. 若 region `1.37x` 无法传播，直接归因 host/receipt/allocation/optimizer crossing；
5. 只有形成 same-solver 性能结论后，再做一次合并外审。

在完成 live bridge 之前，不启动新的外审 exchange。
