# FSG4/B4-B1a Five-Fresh Capture Sufficiency 内部关闭

日期：2026-08-18
状态：`VALIDATED-B4-B1A-FIVE-FRESH-CAPTURE-SUFFICIENCY`

## 正式证据

- source=`4a174235f127f66608a738a3f1a5dee336d719d1`；
- artifact=`artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1`；
- 5 个独立 CUDA worker / 10 captures；
- amendment comparisons=`90 tensors / 63,645 elements`；
- maximum amendment absolute difference=`0.0`，sign exact；
- manifest hash=`67ace9e4a28c84306ee881a41aad1f16d9eddb7e80471ad76b030d691d9b25f6`；
- protocol hash=`a28d465274ff25ea704a21f376c45700f731983c0726e06aae3d13d424013406`；
- summary hash=`38db6fc1380630bdfaad815c44d64cb0a919a4f8c5c26cb85bd3070c91b42738`；
- B4-B0 frozen source identity hash=
  `05b926ac8fc70f03ce6bd08a34b61ef6bf81cb27e02b019c0cb42c2c590c3e9d`；
- 8/8 outer-resigned完整性用例拒绝，report hash=
  `930f11fd1d9dca9d21f8144090c01a74ef11867fa127b87ba364cc08fcdc382f`；
- related=`30 passed`；full=`1382 passed, 3 skipped, 6 warnings`；
- Black、scoped Mypy、Pylint=`10.00/10`、diff、DocOps lint通过。

## 关闭范围

本轮只关闭 reference 输入充分性：incoming/operator bias、region output adjoints、sparse α/β
mapping raw、logical shape/presence/Conv attrs 已在 5 fresh 中稳定并可 root replay。它不证明 typed
IR 或 pure-PyTorch numerical reference 已实现，也不证明 production gradients 已由独立局部公式
重算。

协调一致改写全部run的动态bias/adjoint并全链重签，仍需下一阶段numerical reference代数重算拒绝；
该限制已写入正式 integrity report。下一步开放 B4-B1 typed IR + pure-PyTorch reference 实现；
B4-B2/CUDA/TIR、performance、memory、ASPLOS-ready继续关闭。
