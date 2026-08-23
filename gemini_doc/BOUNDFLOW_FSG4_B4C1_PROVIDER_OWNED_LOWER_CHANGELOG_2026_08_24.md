---
status: implemented-pending-six-fresh
updated: 2026-08-24T11:15:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4c1-provider-owned-lower
stage: s01
---

# FSG4/B4-C1 Provider-owned Lower Changelog

## 结论边界

B4-C1 已把 P-anchor 的 lower ReLU→Conv 所有权移到 CIBC TIR provider：production CROWN 在
provider admission 后只执行 native upper side，lower side不再先算 native 再走 TIR。当前状态只是
`IMPLEMENTED-PENDING-SIX-FRESH`；单 worker 探针语义通过但累计 core 约为 `0.95x`，尚无性能 claim。

## 代码改动

- `crown_ibp.py` 新增 upper-only ReLU 路径，并在 P-anchor Conv 前查询 provider ownership；
- exact-call observer 新增直接 `provide_affine_output`，bridge/provider 互斥且 fail closed；
- TIR forward 改成 4-lane cooperative reduction，仍保持1 forward + 1 backward；
- 10次 optimizer evaluation 复用 plan-owned output/gradient buffer及稳定 DLPack view；
- receipt 新增 provider ownership、buffer reuse和DLPack cache计数；
- cumulative artifact runner支持独立的 `provider-owned-lower` 模式，同时保持历史 B4-C0 replay。

## Correctness

- 单 worker 30组 terminal lower、全部 α/β：allclose/sign exact；
- maximum absolute difference=`7.152557373046875e-07`；
- receipt：10/9 forward/backward、provider-owned=10、bridge=0、buffer reuse=9、
  DLPack cache hit≥63、fallback/eager=0。

## 性能诊断

三个先后探针均显示累计 candidate没有越过 B3：约`0.947x—0.963x`。TIR kernel本身很短：一次
完整 optimizer profile中10个forward合计约`0.221 ms`、9个backward合计约`0.066 ms`。因此失败
不来自 CIBC 算术 kernel。

更关键的是：此前`4.90x`局部对照在 observer 处调用`to_dense()`，强制 native
`SignSplit(Conv2d(...))`立即物化；真实 production baseline保留该算子树并推迟/组合物化。因此局部
PyTorch分母包含了 production原本不会在该边界支付的 eager materialization，不能外推到累计 core。
这是为什么“算子能快几十倍”与“求解器没变快”可以同时成立。

## 下一步

先用6 fresh/180 groups正式关闭 B4-C1。若仍低于1.0，则该单 anchor以
`VALIDATED-NO-GO-B4-C1-MATERIALIZATION-FRONTIER`关闭，不再围绕同一小 region调参；转向 B4-C2：
在真实 materialization frontier接管更大的 lower operator tree，并覆盖 optimizer、KFSB与全部
Linear/Conv/residual fanout。B4-D whole-core/query timing在累计 coverage前仍关闭。
