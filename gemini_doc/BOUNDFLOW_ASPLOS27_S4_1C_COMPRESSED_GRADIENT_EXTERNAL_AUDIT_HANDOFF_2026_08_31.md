# S4-1C compressed gradient 与 terminal lA 外审交接

date: 2026-08-31
executor-verdict: implementation-correctness-candidate
requested-audit-verdict: approve / revise / reject
performance-claimed: false
source-commits: dcbfe80,7110437,82a928a,ef8d704

## 1. 审计任务

请不要采信本文汇总数字。独立阅读冻结施工合同、四个交付文件，现场运行GPU测试，并判断是否足以关闭
`VALIDATED-S4-1C-COMPRESSED-GRADIENT`。本轮只审计single-evaluation dα/dβ、Pass C顺序、V/lA alias、
terminal lease和fail-closed边界；不审计optimizer trajectory或性能。

施工合同：
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_1C_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`。

## 2. 冻结源码范围

1. `boundflow/backends/tvm/asplos27_s4_compressed_gradient.py`
2. `boundflow/runtime/asplos27_s4_gradient_emitters.py`
3. `tests/test_asplos27_s4_compressed_gradient.py`
4. `tests/test_asplos27_s4_gradient_phase.py`

实现提交依次为`dcbfe80`、`7110437`、`82a928a`、`ef8d704`。后续文档提交不应改变上述四文件blob。

## 3. Acceptance criteria

### AC1：范围与门禁顺序

- S4-1B external audit已approved并closed后才出现本批代码；
- 交付文件没有optimizer mutation、timing、performance或same-solver路径；
- 施工包的公式、13-symbol、17/23-action和110 descriptor合同未事后修改。

### AC2：TIR数学与非法输入

- 对六个dα实例逐项推导并用独立公式重算：
  `upstream[d,s] * A[d,s,f] * V[d,s,f]`；
- 只在lower方向、ambiguous且A>=0时非零；alpha闭区间0/1合法；stable或A<0产生+0；
- 检查A/V/bounds/alpha/upstream nonfinite、lower>upper、alpha越界和index越界都产生bits=
  `0x7fc00000`，invalid index在read前被clamp；
- dβ独立核对location=`17,17,31,17,17,31`、sign=`1,1,1,-1,-1,-1`及公式；非法location/sign
  不被静默接受。

### AC3：module与ABI identity

- unscheduled/scheduled module各恰13个PrimFunc：6 dα + 1 dβ + 6 copy；
- dα argument occurrence=48、dβ=5、总计53；unique emitter view=46；
- 由实际TIR JSON/CUDA source重算三个hash，逐个symbol在device source存在；
- global workspace=0、performance flag=false，identity/hash/symbol/claim篡改fail closed。

### AC4：production 17/23-action顺序

- nonterminal动作表与施工包17项逐字一致；terminal为23项；
- site31必须先dα、再dβ、再copy、再transform；
- site25/site19在对应residual stage1与stage2之间发射；
- action换序、漏项、重复项、default/cross stream、metadata/state漂移在launch或commit前拒绝。

### AC5：ownership与terminal handoff

- 七个gradient直接写S4-1A caller-owned buffer，无dynamic output allocation；
- 六个V/lA view只占一个37,464-element / 149,856-byte physical storage；
- terminal copy发生在该site全部V reader之后，copy后保留spec轴；
- terminal lease只能消费一次，nonterminal不能取得lease；
- warm DLPack construction=0，saved dense A/dense gradient escape/fallback/native shadow全为0。

### AC6：独立production correctness

- 现场从冻结ResNet2B pre-state重建S4-0/1A/1B/1C链；
- 在terminal覆盖前独立保存test-only V oracle，在覆盖后从terminal A、V oracle、bounds、indices、upstream
  以不调用candidate TIR的公式重算六dα和dβ；
- 核对全部gradient shape/sign/numeric，以及terminal shaped view和单storage identity；
- 运行新增专项、S4/R3联合、全量tests及Black/mypy/Pylint/diff。

### AC7：claim边界

- 本轮最多形成S4-1C single-evaluation correctness/ownership claim；
- 明确披露coefficient pass当前只在prepare核对R31原始α/β与S4-1A clone内容相等；S4-1D必须把
  coefficient ABI直接改绑active buffers；
- optimizer 10/9 trajectory、KFSB consumer integration、timing、performance、same-solver、complete-query、
  10x和ASPLOS-ready保持false。

## 4. Executor复现命令

```bash
conda activate boundflow

pytest -q \
  tests/test_asplos27_s4_compressed_gradient.py \
  tests/test_asplos27_s4_gradient_phase.py

pytest -q \
  tests/test_asplos27_s4_mutable_state_admission.py \
  tests/test_asplos27_s4_ordered_buffer_abi.py \
  tests/test_asplos27_s4_ternary_endpoint.py \
  tests/test_asplos27_s4_coefficient_selector_pass.py \
  tests/test_asplos27_s4_six_site_value.py \
  tests/test_asplos27_s4_compressed_gradient.py \
  tests/test_asplos27_s4_gradient_phase.py \
  tests/test_r3_d2b_staged_backward.py

pytest -q tests
mypy <上述四个交付文件>
pylint <上述四个交付文件>
git diff --check
```

Executor当前观测为新增专项`11 passed`、联合`200 passed`、全量`2093 passed, 3 skipped`。三个skip为
既有TVM重复编译规避1项和冻结VNN-COMP checkout缺失2项；请审计方独立复现，不采信摘要。

## 5. 预期审计产出

请按AC1—AC7逐项给PASS/FAIL、blocker/major/minor/info、独立公式或脚本、现场命令和最终verdict。若
approve，请明确是否同意关闭`VALIDATED-S4-1C-COMPRESSED-GRADIENT`，并且是否只开放S4-1D
optimizer/evaluator implementation/correctness（timing继续关闭）。
