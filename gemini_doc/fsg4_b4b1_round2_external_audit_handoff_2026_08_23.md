---
status: ready-for-round2-external-audit
updated: 2026-08-23
type: audit-handoff
topic: boundflow
stage: s01
---

# FSG4/B4-B1 Round 2 外审交接

## 1. 请求判定

请独立审计Round 1的F1/F2是否关闭，并判定是否同意关闭：

`VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE-PENDING-EXTERNAL-AUDIT`

本轮result commit=`b8213e2e31a991df77ad484dfaa760b969fe9bd1`，clean reference source=
`e711e991bed54a16c881a2f2bbeb18d71de3c210`。批准只允许executor关闭本exchange并另行预注册
B4-B2；不得解释为B4-B2/TIR已实现或任何性能、显存、whole-core/query、ASPLOS-ready claim。

## 2. Round 1 findings

- F1 major：receipt允许空/缺失metric及错误incoming gradient presence flag；
- F2 major：execution context丢失调用方deterministic warn/debug mode。

请不要采信executor摘要，直接从源码、v3 artifact和测试独立复核。

## 3. F1 acceptance criteria

独立确认`DifferentiableLowerReferenceReceiptV1.validate()`从IR tensor contracts导出精确metric
inventory，并同时绑定：

1. S/P共同五项、S native-beta gradient、P incoming-A gradient的精确名称/数量/排序；
2. 每项`element_count == prod(contract.shape)`；
3. 每项`production_hash == instance.input_tensor_hash_map[target_contract]`；
4. beta/incoming presence flag与IR gradient ownership一致；
5. reference/production digest必须为小写SHA-256；
6. `semantic_passed`仍由完整metric集合的allclose/sign exact导出。

至少独立执行以下负例于S和P：空metrics、删一项、翻转incoming flag、替换production target hash、
替换element count；均须fail closed，且不得仅依赖artifact外层digest。

## 4. F2 acceptance criteria

独立设置调用方deterministic debug mode为0/1/2，分别验证正常退出和context内异常退出：

- context内固定threads=1、debug mode=2、float32 precision=highest、MKLDNN=false；
- context外threads、debug/warn mode、precision、MKLDNN逐项恢复原值；
- 尤其mode=1/warn-only不得被永久升级为mode=2/hard-error。

确认protocol精确冻结：

- `torch_deterministic_debug_mode=2`；
- `torch_deterministic_state_restore=exact-debug-mode-v1`；
- `receipt_metric_inventory=exact-ir-contract-target-v1`。

## 5. v3 provenance与数值门禁

正式证据：

- capture raw：`artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1`；
- v3：`artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v3`；
- integrity：`artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v3-integrity-report.json`。

冻结hash：

- manifest=`2f8a1ffde0f99777e0ab6d9dddb1042c2f7f6c71e57882d141035553475e4e3f`；
- protocol=`b95bc20c8dcaef8635741842b85d4d0bf9e41c9592c60896677907cd96914baf`；
- summary=`753a9558a7c36cb89f02963dcd08fc8e76fdfcd415f7dc5d969eea77dffc7a0b`；
- integrity report=`50a12f577d60a8bf115ee8c40b248f88ecd451715a4b0b4a2f420dedc4aec964`。

独立重算并核对5 runs、10 captures、60 metrics、196,380 elements、max abs diff=
`6.109476089477539e-07`、allclose/sign exact，以及S/P gradient ownership。v1和v2必须由新protocol
fail closed拒绝，v3必须root replay逐字节通过。正式两类all-run完整性负例须`2/2 rejected`；
建议再新增一类receipt inventory或execution-policy独立负例。

## 6. 回归与静态门禁

Executor声明值：

- targeted：`32 passed`；
- B3/B4 related：`140 passed`；
- full（RTX 4060 Laptop GPU）：`1414 passed, 3 skipped, 6 warnings`；
- Black：3 files unchanged；
- scoped Mypy：PASS；
- scoped Pylint：`10.00/10`；
- diff、`dol exchange validate`、`dol lint --soft`：PASS。

请独立运行并记录实际数字。三个声明skip应仅为allow-no-TVM重复编译与两项冻结VNN-COMP
checkout不可用，不得包含B4-B1/CUDA路径。

## 7. 输出格式

请输出verdict、F1/F2 closed/open、AC逐项PASS/FAIL、独立数值、findings分级、不可现场复核项，
并明确是否允许executor关闭exchange以及批准后是否只开放B4-B2预注册。
