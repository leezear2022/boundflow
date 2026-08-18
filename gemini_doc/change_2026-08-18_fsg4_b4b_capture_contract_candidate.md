# FSG4/B4-B0 Typed Production-Region Capture Contract 候选记录

日期：2026-08-18

状态：`IMPLEMENTED-B4-B0-CAPTURE-CONTRACT-PENDING-LIVE-HOOK`

## 改动

- 新增`boundflow/runtime/fsg4_b4b_production_region_capture.py`；
- 冻结`Gemm_14`的active-beta语义锚点与`Conv_8`候选性能锚点；
- 新建live CUDA tensor→immutable CPU raw payload合同，保留shape/dtype/device/stride/requires-grad/
  content digest；
- capture必须是optimizer evaluation 0、provider/fallback/eager-backward-fallback全0；
- 对Conv强制weight shape、stride/padding/dilation/groups；
- 显式分离production compressed alpha/beta映射源与native dense alpha/beta、
  `relu_pre_add_coeff_l`及native gradients。production compressed state不被伪造为exact-region leaf。
- incoming A只在production raw本身`requires_grad=true`时要求gradient；顶层锚点不伪造
  production gradient，custom-backward的incoming-A gradient由后续micro requires-grad clone验证。

## 负向门禁

新测试覆盖：非0 evaluation ordinal、缺native beta gradient、active-beta改为空、Conv attrs不完整、
tensor内容被篡改、CPU placeholder，以及两个正向锚点。

## 验证

- `pytest -q tests/test_fsg4_b4b_production_region_capture.py`：`10 passed`；
- B4-B/PR-12/B4-A/B3 fixed related：`45 passed`；
- incoming-A所有权负向测试加入后fixed related：`46 passed`；
- full：`1366 passed, 3 skipped, 6 warnings in 443.94s`；
- Mypy explicit package bases：PASS；
- Pylint：`10.00/10`；
- Black：PASS。

## 边界与下一步

本次只建立typed capture substrate，尚未接入live solver，没有5 fresh artifact、gradient parity、
TIR或performance claim。下一唯一动作是将该合同接到optimizer evaluation 0的显式opt-in
observer，在timed region之外生成两锚点raw capture。
