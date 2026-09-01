---
status: closed-approved
updated: 2026-08-23T05:45:00Z
type: audit-handoff
topic: boundflow
slug: fsg4-b4b2-b2-1-dense-linear-external-audit
stage: s01
---

# FSG4/B4-B2 B2-1 Dense Linear TIR External Audit Handoff

> **2026-08-23 外审关闭**：`APPROVE`，0 blocker/0 major；本交接中B4-B related=`76`
> 已被审计方现场更正为`77 passed`。当前只开放B2-2，timing/P-anchor/B2-4/B2-5/
> B4-B3仍关闭。

## 1. 审计对象与冻结点

- branch=`feat/rvir-v4-production-state-ownership-v1`；
- source=`eb74e45`；base=`09c559d`（B2-0外审关闭）；
- preregistration=`57be636`；B2-0 implementation=`712ca03`；
- 本轮状态上限=
  `VALIDATED-B4-B2-B2-1-DENSE-LINEAR-CORRECTNESS-PENDING-EXTERNAL-AUDIT`；
- 本轮无artifact写入、无timing、无B2-2 sparse-source、无production exact-call改动。

审计必须以`eb74e45`为代码冻结点。后续纯审计文档提交不得替换source；若任一产品代码变化，必须
重新声明source并重跑全部门禁。

## 2. 变更范围

新增：

- `boundflow/ir/differentiable_lower_dense_linear_tir.py`；
- `boundflow/backends/tvm/differentiable_lower_dense_linear.py`；
- `boundflow/runtime/fsg4_b4b2_dense_linear_tir.py`；
- `scripts/run_fsg4_b4b2_dense_linear_tir_correctness.py`；
- `tests/test_fsg4_b4b2_dense_linear_tir.py`；
- 本轮changelog与权威状态文档。

修改B2-0文件仅用于关闭外审minor：

- `boundflow/runtime/fsg4_b4b2_identity_tir.py`：fallback/eager从receipt硬编码改为executor计数器；
- `tests/test_fsg4_b4b2_identity_tir.py`：新增计数器拒绝测试。

禁止把新增dense ABI解释为production集成。`git diff 09c559d eb74e45`不得出现B2-2、Conv、timing、
same-solver、optimizer activation或旧production执行路径改动。

## 3. 不可信执行方摘要

以下数字只作为待独立复核声明，不得直接采信：

- 5 fresh S captures；
- 20 metrics / 36,750 elements；
- max abs diff=`8.642673492431641e-07`；
- allclose/sign exact=`true/true`；
- cache=`miss,hit,hit,hit,hit`；
- 每run forward/backward=`1/1`，fallback/eager=`0/0`；
- template=`d96bb8d62eb2e112e4f9ac5e98bc971cb41122cd97273ebb3fc1c4fc5c0a0be4`；
- schedule=`989c3eae7fcefed3a6399b000c51eb222c5e5ba2a31a220ef42db5d86ca5de4b`；
- module receipt=`e99121435e5db022c02f1d1610ffb9d4048397e09168f91f6857e425ad80801a`；
- targeted=`23 passed`，B4-B related=`77 passed`（外审现场更正），full=
  `1437 passed, 3 skipped, 6 warnings`。

## 4. Acceptance criteria

### AC1：顺序、source与scope

独立确认：

1. B2-0外审批准先于B2-1代码；
2. source/base/prereg commit存在且branch/origin一致；
3. `09c559d..eb74e45`只包含B2-1与两项已披露minor关闭；
4. 无B2-2 sparse ABI、P-anchor/Conv、timing/performance、production path或claim越级；
5. `.docops/s.md`只开放外审。

### AC2：first-class编译/执行身份

不信任现成hash，独立从对象与TVM module重算：

1. Template绑定S-anchor `semantic-active-beta-gemm-14`、`[6,1,100]`、`[100,1024]`、
   active beta、operator bias、mapping/layout/operator attrs、sm_89；
2. Instance对每run绑定11个输入/adjoint tensor hash，动态值不进入compile key；
3. Schedule仅`dense-linear-serial-reduction-v1`、128 threads、candidate ordinal=0，明确保留dense
   workspace且`performance_admitted=false`；
4. Module receipt从unscheduled/scheduled TIR、device source、双symbol、TVM/FFI commit独立重算；
5. Launch receipt绑定11 inputs、4 outputs/gradients、23个DLPack pointer、stream/cache/launch/counter；
6. Template/Instance/Schedule/Module/Launch canonical round-trip与stable hash一致；
7. 合法重签但改变anchor/shape/ABI/schedule/toolchain/symbol/inventory的对象必须fail closed。

### AC3：五份raw数值与离散导数所有权

审计方必须读取：

`artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1/run_00.pt`至`run_04.pt`，每份只选
S-anchor capture。不得只调用执行方summary函数；至少用独立PyTorch表达式或标准库控制脚本重算：

1. lower sign selection与upper slope/intercept；
2. `clamp(alpha,0,1)` lower slope；
3. active beta=`-native_beta*dense_split_sign`；
4. `relu_lower_a @ weight`与operator bias reduction；
5. 真实output A/bias adjoint传播至native α/β gradient；
6. incoming A执行前后hash不变、incoming A gradient absent；
7. 4 metrics/run、元素总数36,750；全部finite、`atol=rtol=2e-4`、sign exact；
8. 单独核对α=0与α=1的clamp导数包含端点，以及`A==0`选择lower branch但α VJP因A为0而为0；
9. S dense β gradient present且每run恰6个非零位置；P-anchor必须被admission拒绝。

### AC4：runtime与失败路径

现场GPU复跑并亲读代码确认：

1. forward/backward各恰一次PackedFunc module launch；
2. 23/23 DLPack round-trip data_ptr exact，所有4 output/gradient不alias输入；
3. 显式非默认current stream与TVM-FFI raw stream双向一致；
4. first run cache miss，后四run hit；动态tensor变化不改变module identity；
5. fallback/eager为executor真实计数器，任何`reject_fallback`调用先计数再raise；
6. higher-order、错误adjoint pointer、nonfinite/dtype/device/layout、重复launch全部fail closed；
7. 在TVM context内部触发missing-symbol异常后，current device、current stream、deterministic
   enabled/debug mode全部原样恢复。

### AC5：replayable runner与claim边界

现场运行`python scripts/run_fsg4_b4b2_dense_linear_tir_correctness.py`，输出必须可重复产生§3三项
receipt hash与数值摘要；但审计结论必须由AC2/AC3独立重算支撑。确认新代码无timing API、
`performance_claimed=false`、`sparse_source_admitted=false`，文档不得外推region fusion、memory、
speedup、B0 parity、whole-query或ASPLOS-ready。

### AC6：测试、静态与DocOps

至少现场运行：

```bash
source /home/lee/miniconda3/etc/profile.d/conda.sh
conda activate boundflow
python -m pytest -q tests/test_fsg4_b4b2_identity_tir.py tests/test_fsg4_b4b2_dense_linear_tir.py
python -m pytest -q tests/test_fsg4_b4b*.py
python -m pytest -q -rs
python -m black --check \
  boundflow/ir/differentiable_lower_dense_linear_tir.py \
  boundflow/backends/tvm/differentiable_lower_dense_linear.py \
  boundflow/runtime/fsg4_b4b2_dense_linear_tir.py \
  scripts/run_fsg4_b4b2_dense_linear_tir_correctness.py \
  tests/test_fsg4_b4b2_dense_linear_tir.py \
  boundflow/runtime/fsg4_b4b2_identity_tir.py \
  tests/test_fsg4_b4b2_identity_tir.py
python -m mypy \
  boundflow/ir/differentiable_lower_dense_linear_tir.py \
  boundflow/backends/tvm/differentiable_lower_dense_linear.py \
  boundflow/runtime/fsg4_b4b2_dense_linear_tir.py \
  scripts/run_fsg4_b4b2_dense_linear_tir_correctness.py \
  boundflow/runtime/fsg4_b4b2_identity_tir.py
python -m pylint \
  boundflow/ir/differentiable_lower_dense_linear_tir.py \
  boundflow/backends/tvm/differentiable_lower_dense_linear.py \
  boundflow/runtime/fsg4_b4b2_dense_linear_tir.py \
  scripts/run_fsg4_b4b2_dense_linear_tir_correctness.py \
  tests/test_fsg4_b4b2_dense_linear_tir.py \
  boundflow/runtime/fsg4_b4b2_identity_tir.py \
  tests/test_fsg4_b4b2_identity_tir.py
bash scripts/rebuild_tvm.sh
python /home/lee/.codex/plugins/cache/personal/docops-logic/0.2.0+codex.20260802165548/scripts/dol.py lint --soft
```

3个skip必须逐项核对为既有环境边界，不能只报总数。

## 5. 已知边界

- dense ABI在TIR内显式保留`output_bias_delta`、`adjoint_matmul`、`adjoint_relu` workspace；
- five fresh raw为确定性重复capture，五份hash相同不等于五份process性能证据；本阶段不计时；
- raw stdout未冻结artifact，formal raw/replay/tamper仍属于B2-5；
- 当前只覆盖S Linear，不覆盖P Conv、sparse-source mapping kernel或compressed gradient projection；
- execution receipt是correctness evidence，不是production activation receipt。

## 6. 审计输出格式

报告请放在`gemini_doc/`，至少包含：

1. verdict：`approve`或`request_changes`；
2. blocker/major/minor/info逐项finding；
3. AC1—AC6逐条PASS/FAIL与独立证据；
4. 独立重算的5-run/20-metric/36,750-element数值；
5. 现场GPU、targeted/related/full/static/DocOps结果；
6. 对两条B2-0 minor是否关闭的明确判断；
7. claim边界与下一动作判定。

只有`approve`且无blocker/major，才允许关闭B2-1并开放B2-2 S-anchor sparse-source fused
forward/backward。任何结果都不开放timing、P-anchor、B2-4/B2-5或B4-B3。
