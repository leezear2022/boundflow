---
status: validated-s2-4x-canonical-crown
date: 2026-08-28
type: formal-closure
topic: boundflow
slug: asplos27-s2-coarse-crown-custom-vjp
source: d9582b552348c534dc5fb039231496e21c9f9b4c
artifact: artifacts/asplos27-s2-crown-pipeline/resnet2b-p-anchor-v2
performance-claimed: false
---

# ASPLOS’27 S2 coarse CROWN + custom VJP formal closure

## 1. 判定

S2按预注册协议关闭为：

```text
VALIDATED-S2-4X-CANONICAL-CROWN
```

该状态只说明在冻结ResNet2B P-anchor、6 domains、一次lower evaluation + compressed α VJP的同scope
比较中，canonical BoundFlow candidate达到冻结的4×门槛。它不是optimizer、solver、query或ASPLOS总性能
claim，所有artifact仍保存`performance_claimed=false`。

## 2. 实现闭环

```text
production plan + bounded lifetime trace
  → active-β coefficient/sign wavefront（two-slot arena）
  → selected input TIR
  → cuDNN Conv0
  → selected ReLU17 TIR
  → cuDNN Conv2
  → selected ReLU19 TIR
  → cuDNN Conv4 + shortcut Conv5
  → selected ReLU23 TIR
  → cuDNN Conv8
  → recompute-A26 + compressed-gradient TIR
  → compact lower + compressed dα
```

关键结构事实：

- 五个Conv调用全部进入TVM cuDNN；Conv4/Conv8同签名共享编译函数，因此是4个partition functions、
  5个call sites；
- input/ReLU17/19/23 selection为4个verification-specific scheduled TIR；
- fixed forward wavefront与selected-value chain各一次CUDA Graph replay；
- 28个argument views只在prepare阶段DLPack绑定，warm construction=`0`；
- 复用R3-D2B plan、trace、state、active β与two-slot arena；没有新造solver IR；
- saved dense A=`0`，saved autograd history=`false`，fallback/eager/native shadow=`0`。

## 3. Formal protocol

三方定义：

- `N`：原生PyTorch forward + autograd VJP；
- `D`：历史direct D2B compiled custom VJP；
- `P`：S2 canonical prepared direct custom VJP。

执行六个fresh process，顺序固定为`NDP/NPD/DNP/DPN/PND/PDN`；每进程5 warmup groups、30 measured
groups；同一non-default stream，调用前后device boundary sync。共保存540个原始latency样本。

| order | N ms | D ms | P ms | N/P | D/P |
|---|---:|---:|---:|---:|---:|
| NDP | 8.935410 | 4.754501 | 1.920764 | 4.652010× | 2.475318× |
| NPD | 9.049729 | 5.459606 | 2.238135 | 4.043425× | 2.439356× |
| DNP | 9.114046 | 5.320129 | 2.227510 | 4.091586× | 2.388375× |
| DPN | 8.918558 | 4.766214 | 1.930262 | 4.620389× | 2.469206× |
| PND | 9.047216 | 5.061914 | 1.945699 | 4.649852× | 2.601591× |
| PDN | 8.709091 | 5.994125 | 2.459640 | 3.540799× | 2.436993× |

聚合结果：

- P/N geomean=`4.24538196457207x`（门槛`>=4.00x`）；
- P/N worst=`3.540798856743263x`（门槛`>=3.50x`）；
- P/D geomean=`2.4676101727573547x`（门槛`>=0.90x`）；
- canonical cold prepare geomean=`1.61270395874274 s`，单独披露，不混入warm headline；
- warm dynamic allocated/reserved=`0/0 bytes`（PyTorch allocator口径）。

## 4. Correctness与证据完整性

- lower max absolute diff=`3.0994415283203125e-06 <= 2e-4`；
- compressed dα max absolute diff=`6.146728992462158e-08 <= 2e-4`；
- lower与gradient sign全部exact；
- source commit=`d9582b552348c534dc5fb039231496e21c9f9b4c`；protocol逐blob绑定7个关键文件；
- source capture/model、plan/trace、source/partitioned/lowered Relax、device sources和receipt hash全链绑定；
- raw-only replay PASS，summary hash=
  `694c011ae80fa4131c2fcc3112bfcd75ae1ab4e502763797662e6fb2755482e4`；
- 10类inner/outer-resigned攻击全部拒绝，包括latency+summary重算、lower、gradient、cuDNN call count、
  forward replay、active β、performance claim、order、plan owner与source identity。

## 5. 验证

- S2专项：`7 passed`；
- 全量：`1876 passed, 3 skipped`；三个skip均为既有TVM/VNN-COMP环境边界；
- black：clean；
- mypy：2个production文件clean；
- pylint：6个相关文件`10.00/10`；
- artifact replay与tamper：PASS / `10/10 rejected`。

## 6. 尚未成立

- 10 evaluation / 9 optimizer mutation trajectory性能；
- RVIR exact-call same-solver替换；
- complete-query、queue、TTV/solved；
- 第二model family或held-out property；
- 总体10×和ASPLOS-ready。

因此下一动作不是继续宣布更大性能，而是只写S3预注册：把同一个prepared direct VJP接回冻结的10/9
trajectory，保持α初始化、Adam/scheduler、clamp、terminal export与state mutation顺序，重新做N/D/P formal。
