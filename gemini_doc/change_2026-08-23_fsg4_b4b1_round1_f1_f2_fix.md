---
status: validated-pending-round2-external-audit
updated: 2026-08-23
type: change-record
topic: boundflow
stage: s01
---

# FSG4/B4-B1 Round 1 F1/F2 修复记录

## 结论

B4-B1 Round 1 外审正式判定为 `request_changes`，包含两个 major：

- F1：typed receipt 未精确绑定必须存在的 metric、gradient target 与 presence flag；
- F2：reference execution context 未原样恢复调用方的 PyTorch deterministic
  warn/debug mode。

两个 finding 均接受并已实现修复；先前会话的 `.git` 只读阻塞已解除。修复仍需以本次 clean
source commit生成provenance-bound v3 artifact并提交Round 2，因此本记录不主张B4-B1已经
外审关闭。

## 修改

### F1：receipt exact inventory 与 target binding

`DifferentiableLowerReferenceReceiptV1.validate()` 现在从 typed IR tensor contracts 导出精确
metric 清单：共同五项、S 的 native-beta gradient、P 的 incoming-A gradient。验证同时要求：

- metric 名称、数量与排序完全一致，空清单或缺项 fail closed；
- `element_count` 等于目标 contract shape 的元素数；
- `production_hash` 等于 instance 中对应 production value/gradient target 的 digest；
- beta/incoming presence flags 与 IR gradient ownership 完全一致；
- metric digest 为小写 SHA-256。

新增负向测试覆盖空清单、缺项、presence flag 翻转、target digest 替换和元素数替换。

### F2：完整 execution-policy 恢复

reference runner 现在保存 `torch.get_deterministic_debug_mode()`，在 context 内冻结为 error
mode 2，退出时恢复原始 mode，而不是只恢复 deterministic boolean。协议新增：

- `torch_deterministic_debug_mode=2`；
- `torch_deterministic_state_restore=exact-debug-mode-v1`；
- `receipt_metric_inventory=exact-ir-contract-target-v1`。

新增参数化测试覆盖初始 mode 0/1/2，以及正常退出和异常退出；同时核对 threads、float32 matmul
precision 和 MKLDNN 均原样恢复。旧 v2 因缺少新协议字段明确 fail closed。

## 当前验证

- clean source：`e711e991bed54a16c881a2f2bbeb18d71de3c210`；
- v3 manifest：`2f8a1ffde0f99777e0ab6d9dddb1042c2f7f6c71e57882d141035553475e4e3f`；
- v3 protocol：`b95bc20c8dcaef8635741842b85d4d0bf9e41c9592c60896677907cd96914baf`；
- v3 summary：`753a9558a7c36cb89f02963dcd08fc8e76fdfcd415f7dc5d969eea77dffc7a0b`；
- v3 integrity report：`50a12f577d60a8bf115ee8c40b248f88ecd451715a4b0b4a2f420dedc4aec964`，
  `2/2 rejected`；
- v3 root replay：5 runs、10 captures、60 metrics、196,380 elements、max diff=
  `6.109476089477539e-07`、sign exact；
- targeted：`32 passed`；
- B3/B4 related：`127 passed, 12 skipped`；
- full（完整激活`boundflow`环境）：`1365 passed, 51 skipped, 7 warnings`；
- scoped Mypy：PASS；
- scoped Pylint：`10.00/10`；
- `git diff --check`：PASS。

以上为修复工作树的候选验证，不替代 clean-source v3 artifact 或外部 Round 2 审计。首次直接
调用conda解释器的full在collection阶段因未触发activation hook、TVM不可见而停止；完整
`conda activate boundflow`后重跑全绿，已明确区分环境调用错误与代码回归。

## 下一步与边界

1. 完成最终 related/full/static/DocOps validation；
2. 对 F1/F2 各执行一次 `dol exchange respond`，然后 delivery Round 2；
3. 只有 Round 2 独立批准并关闭 exchange 后，才可另行预注册 B4-B2。

B4-B2、CUDA/TIR、performance、memory、whole-core/query speedup 与 ASPLOS-ready 继续关闭。
