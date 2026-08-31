# S4-1B 六站点 production correctness 外审交接

date: 2026-08-31
executor-verdict: implementation-correctness-candidate
requested-audit-verdict: approve / revise / reject
performance-claimed: false
source-commit: 760fa0d

## 1. 外审任务

请不要采信本交接中的汇总数字，独立核对源码、运行GPU专项，并判断本批是否足以关闭
`VALIDATED-S4-1B-SIX-SITE-VALUE`。审计范围只含Pass A selector生成、Pass B六站点value图、arena/stream/
receipt ownership和correctness；不审计S4-1C梯度、optimizer或性能。

## 2. 冻结源码范围

只审计以下五个交付文件及其固定实现提交`760fa0d`：

1. `boundflow/backends/tvm/asplos27_s4_six_site_value.py`
2. `boundflow/runtime/asplos27_s4_coefficient_selector_pass.py`
3. `boundflow/runtime/asplos27_s4_six_site_value.py`
4. `tests/test_asplos27_s4_coefficient_selector_pass.py`
5. `tests/test_asplos27_s4_six_site_value.py`

施工合同：
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`。

## 3. 建议 acceptance criteria

### AC1：范围与顺序

- S4-1B0外审已关闭后才出现本批实现；
- 交付文件没有S4-1C dα/dβ、optimizer、timing或performance路径；
- construction package的19-action、42+7 ABI和phase顺序未事后修改。

### AC2：Pass A真实生产边界

- 独立读`capture_r31b2_production_selectors_v1`，确认A29在ReLU28前、A26/A20分别位于两个residual
  stage1/stage2之间、Ainput在box concretize前；
- 确认6个selector由预绑定DLPack view的TIR kernel写入，production路径eager pack=`0`；
- endpoint合法值为`{-128,-1,0,1}`，binary合法值为`{-128,0,1}`，nonfinite不被静默映射。

### AC3：Pass B编译图

- 独立从Relax IR重建49参数顺序，确认42 read和7 caller-owned write target；
- 核对6 Conv、1 Gemm、1 ternary input select、5 selected-ReLU、6 persistent copy；
- active α为`[D,W]`，empty β与site31 α/map/sign没有进入Pass B；
- source/partition/lowered/device source hash由实际content重算。

### AC4：ownership与运行时

- selected-input与coefficient arena同storage，rebind前live reader=`0`；
- V17/19/23/25/28/31为单一37,464-element arena连续无洞view；
- 49个DLPack view仅prepare创建，warm view count=`0`，result owner没有无界list；
- current Torch stream、TVM-FFI stream和prepared stream三者一致，default stream拒绝；
- receipt不含raw pointer/Tensor/NDArray/VM对象。

### AC5：独立数值复核

- 现场运行真实冻结ResNet2B测试；
- 不复用TVM结果作为oracle，独立用PyTorch重算V17/V19/V23/V25/V28/V31；
- 核对每槽shape、pointer owner、数值容差和selector原始内容。

### AC6：负向与回归

- 复核phase/order/generation/pointer/alias/hash/count/claim篡改fail closed；
- 运行新专项、S4/R3联合、全量tests以及Black/mypy/Pylint/diff；
- 披露所有skip理由。

### AC7：claim边界

- 只能形成S4-1B value correctness/ownership claim；
- S4-1C、optimizer、timing、performance、same-solver、complete-query、10x、ASPLOS-ready保持false。

## 4. Executor当前复现命令

```bash
conda run -n boundflow pytest -q \
  tests/test_asplos27_s4_coefficient_selector_pass.py \
  tests/test_asplos27_s4_six_site_value.py

conda run -n boundflow pytest -q \
  tests/test_asplos27_s4_mutable_state_admission.py \
  tests/test_asplos27_s4_ordered_buffer_abi.py \
  tests/test_asplos27_s4_ternary_endpoint.py \
  tests/test_asplos27_s4_coefficient_selector_pass.py \
  tests/test_asplos27_s4_six_site_value.py \
  tests/test_r3_d2b_staged_backward.py
```

Executor观测为新增专项`9 passed`、联合`189 passed`、全量`2082 passed, 3 skipped`；请独立复算。
三个skip是既有环境/重复编译边界，不采信本摘要，请用`-rs`自行核对。

## 5. 预期审计产出

请按AC1—AC7逐条给PASS/FAIL、blocker/major/minor/info、可复现命令、独立重算数字和最终verdict；若
approve，请明确是否同意升级为`VALIDATED-S4-1B-SIX-SITE-VALUE`，以及是否只开放S4-1C
implementation/correctness。
