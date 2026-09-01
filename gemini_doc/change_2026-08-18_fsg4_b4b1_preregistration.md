# FSG4/B4-B1 预注册变更记录

日期：2026-08-18
状态：`PREREGISTERED-B4-B1-TYPED-PYTORCH-REFERENCE`

本轮无产品代码改动。新增 B4-B1 typed pure-PyTorch reference 预注册，依据 B4-B0 v2 raw
识别并量化 capture sufficiency 缺口：现有输入可将 output A 重建至约 `3e-8`，但缺少 incoming
bias、operator bias 与 region output adjoints，不能独立重放完整 bias 和 production gradients。

计划因此新增 B4-B1a capture amendment，冻结 sparse α/β raw、bias ownership、output adjoints、
typed IR、reference、five-fresh correctness/gradient 与 fail-closed 门禁。B4-B2、CUDA/TIR、
performance、memory 与 ASPLOS-ready 继续关闭。

主计划：
`gemini_doc/BOUNDFLOW_FSG4_B4B1_TYPED_PYTORCH_REFERENCE_PREREGISTRATION_2026_08_18.md`。
