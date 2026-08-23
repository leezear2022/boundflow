---
status: externally-approved-b2-2-b2-3-open
updated: 2026-08-23T08:45:00Z
type: external-audit
topic: boundflow
slug: external-audit-b4b2-b2-2-sparse-linear-tir
stage: s01
---

# FSG4/B4-B2 B2-2 Sparse-source Linear TIR 独立外部审计报告

- 审计对象：B4-B2 路线 B2-2 切片（S-anchor sparse-source fused Linear TIR
  forward/backward correctness）。
- 分支：`feat/rvir-v4-production-state-ownership-v1`；审计 HEAD=`7a3f5f4`（与 origin
  一致，`git status -sb` 无 ahead/behind）；实现提交=`8bd1db2`；base=`a6ca7f8`。
- 审计方未采信交接文档任何数字，全部现场独立复核。

## 总体 Verdict

**APPROVE**。0 blocker、0 major、0 minor、2 info。同意关闭 B2-2 并开放
**B2-3 P-anchor Conv dense correctness**（仅此一项）；timing、B2-4/B2-5、B4-B3
仍全部关闭。

## 逐项结论（对应交接 AC1–AC7 与审计清单 1–8）

### 1. git/范围：PASS

- `git show --stat 8bd1db2`：仅 5 个新增代码文件
  （`boundflow/ir/differentiable_lower_sparse_linear_tir.py` 898 行、
  `boundflow/backends/tvm/differentiable_lower_sparse_linear.py` 420 行、
  `boundflow/runtime/fsg4_b4b2_sparse_linear_tir.py` 903 行、
  `scripts/run_fsg4_b4b2_sparse_linear_tir_correctness.py` 131 行、
  `tests/test_fsg4_b4b2_sparse_linear_tir.py` 435 行）+ 状态/变更文档；
  `git diff --stat a6ca7f8 8bd1db2 -- boundflow/3rdparty` 为空（0 行），vendored TVM
  未动，故本轮无需 TVM rebuild（changelog 称"增量树无需重编译"，与事实一致）。
- `7a3f5f4` 只加交接文档与 DocOps 事件。
- 预注册文档本轮 16 行更新与总计划 11 行更新均为**追加式状态块**
  （`git show 8bd1db2 -- <path>` 逐行核对），B2-2 门禁条文
  （compressed α/β 直接 TIR、compressed gradient 投影、禁止 dense workspace、
  失败即 NO-GO）无任何事后改动。
- 无 Conv/P-anchor/timing/production path/optimizer 改动；新代码 grep
  无 `perf_counter`/`time.`/`speedup` 计时 API。

### 2. 数学正确性：PASS（重点，含独立 float64 重算）

亲读 `boundflow/backends/tvm/differentiable_lower_sparse_linear.py` 全文并独立推导：

- **compressed α 布局/索引**：`_compressed_alpha_value`（backend 行 42–50）以
  27 项 if-then-else 链把 feature index 映射到 `compressed_alpha[d, ordinal]`；
  未映射 feature 取常量 0，与 oracle `_reconstruct_alpha`
  （`fsg4_b4b1_pytorch_reference.py` 行 190–254，dense zeros + index_put）语义一致——
  未映射处 native α=0。**无 dense 物化**。
- **β pre-add**：`_beta_pre_add`（行 86–93）仅在每 domain 冻结 location 上给
  `-compressed_beta[d,0]*sign[d]`，与 oracle `_reconstruct_beta` 的
  `pre_add = -dense*split` 一致。
- **forward**：slope 选择（incoming≥0→lower_slope，否则 upper_slope）、
  ambiguous intercept、output_lower_a=relu@W、
  output_bias=incoming_bias+Σ(incoming·intercept)+Σ(relu·op_bias)，
  与 B4-B1 pure-PyTorch oracle 行 356–406 逐项对应。
- **backward compressed gradients**：
  `compressed_alpha_gradient[d,k] = Σ_s adjoint_relu[d,s,feat_k]·incoming[d,s,feat_k]`
  （gate：incoming≥0 ∧ lower<0 ∧ upper>0 ∧ 0≤α≤1），与 oracle 对
  `native_alpha.clamp(0,1)` 的 autograd VJP 一致；`compressed_beta_gradient[d] =
  -Σ_s adjoint_relu[d,s,loc_d]·sign_d]`（`beta*0` 项仅为绑定 beta buffer，数值恒为 +0）。
  27 个 feature index 经 template.validate 强制严格递增且唯一（IR 行 178–184），
  每个 source 元素被覆盖且恰一次。
- **reference 独立性**：runner 的 reference 来自
  `run_b4b1_pytorch_reference_v1`（纯 PyTorch + autograd，B4-B1 已与 production
  对账），非 TIR 自比；`references` 字典直接取 oracle 输出与 native 梯度 gather。
- **clamp 端点语义**：现场实测 PyTorch `clamp(0,1)` 在 x=0/1 处梯度=1（含端点），
  与 TIR 条件 `alpha>=0 && alpha<=1` 一致；156/162 个 compressed α 值恰在端点，
  此约定是梯度主路径，TIR/oracle 两端一致。
- **审计方独立 float64 重算**（`/tmp/b2_2_independent_audit.py`，不用执行方 metric
  helper、不用 autograd，直接写闭合公式）：5 个 raw 的 4 项输出对 TIR 结果
  max diff 分别为 output_lower_a=4.27e-08、output_bias=5.13e-08、
  compressed_alpha_gradient=6.69e-07、compressed_beta_gradient=6.99e-07，
  全部远低于 atol=rtol=2e-4；beta gradient 6/6 nonzero。
- **capture 独立核验**：5 个 `run_0*.pt` 的 S capture 逐一读取：
  production alpha=`[2,1,6,27]`、beta=`[6,1]`；27 个 feature index
  严格递增、唯一、全部落在 [0,100)（实测值
  0,1,3,4,6,13,17,20,24,27,29,30,31,32,42,45,46,58,64,65,75,78,86,88,89,90,93）；
  beta location=[17,17,31,17,17,31] 在界内，sign=[1,1,1,-1,-1,-1]∈{±1}；
  5 个 capture 的 mapping 完全一致，且全部 value tensor 逐位相同
  （确定性重复，交接 §5 已披露）。native oracle 在 unowned 坐标的梯度现场核实恰为 0
  （α、β 各 0 个非零）。

### 3. workspace 禁令：PASS（重点）

- 审计方现场调用 `build_sparse_linear_tir_modules` 重建 scheduled TIR 并直读 script
  （留存 `/tmp/b2_2_scheduled_tir.txt`）：全模块仅 2 个 `T.alloc_buffer`——
  forward 的 `output_bias_delta (6,1)` 与 backward 的 `adjoint_matmul (6,1,100)`；
  `native_alpha`/`native_beta`/`relu_lower_a`/`scaled_a` 计数均=0
  （relu_lower_a 已被 compute_inline 消除；adjoint_relu 同样 inline）。
- forward 签名无 native dense α/β 输入（8 个输入），backward 输出直接是
  `[6,27]`/`[6,1]`，无后置 dense gather；script 中 `[6,100]` buffer 仅为
  preactivation I/O 参数，无任何 `[6,100]` 物化 workspace。
- 检测机制是双层：结构性（primfunc 表达式只读 compressed buffer，不可能物化 dense）
  + receipt 层（scheduled script 子串计数，`build` 与 `compile` 两处 fail-closed
  raise）。子串计数理论上可被"换个名字"绕过，但结构性保证不依赖命名，
  且 Module receipt 的 observed/forbidden 由实际 scheduled TIR 重算并经
  `validate_against` 强制 `forbidden_workspace_count==0`。
- 元素口径：31,590 = 5×(6144+6+162+6)；B2-1 dense 为 36,750 = 5×(6144+6+600+600)，
  差异恰为 native[6,100]→compressed[6,27]/[6,1] 的梯度收缩，符合预期。

### 4. 实测独立复核：PASS

现场重跑 `python scripts/run_fsg4_b4b2_sparse_linear_tir_correctness.py`
（exit=0，RTX 4060 Laptop, sm_89）：

- status=`validated-b2-2-sparse-source-linear-correctness`；
  run/metrics/elements=`5/20/31590`；max diff=`8.642673492431641e-07`；
  allclose/sign exact 全 True；cache=`miss,hit,hit,hit,hit`；
  forward/backward launch 每 run `1/1`；fallback/eager `0/0`；
  每 run projection 六项（α/β mapping exact、α/β numerical、nonzero sign exact、
  unowned native zero exact）全 True。
- 三个 hash 与交接**逐位一致**：template=`adddcb6a…f56f`、
  schedule=`b8fe0a7d…0d57`、module receipt=`7f6ab5cb…2679`
  （template/schedule 另由审计方用 `canonical_tir_hash` 独立重算，一致）。
- DLPack 21/21 与代码口径吻合：forward 8 输入+2 输出=10，backward 9+2=11。
- max diff 与 B2-1 完全相同：已独立解释——5 个 capture 逐位相同且
  output_lower_a 走与 dense 相同的数值路径，对同一 float32 oracle 的最大偏差
  逐位复现属合理。

### 5. 测试质量：PASS

- 11 个测试全部亲读：receipt round-trip + 重签 fail-closed（ABI/mapping 重复、
  location 越界、sign 非法、workspace count、fallback、performance_claimed 篡改）、
  P-anchor capture 拒绝、5-fresh parity + cache 事件、scheduled TIR workspace 断言、
  **dtype/device/nonfinite/range 拒绝专项**（上轮 info 已在 sparse 侧补齐）、
  clamp 端点+A==0 梯度所有权（断言掩码非空且梯度恰为 0）、custom stream 异常后
  device/stream/deterministic 状态恢复、fallback 真实计数、higher-order 拒绝。
- 断言具体（match 具体错误串、精确计数），tamper 路径真实重签后拒绝。
- 现场运行：targeted `34 passed in 14.77s`；B4-B related `88 passed in 23.32s`。

### 6. 测试/静态：PASS

- 全量现场重跑：`1448 passed, 3 skipped, 6 warnings in 466.15s`（exit=0）。
  `-rs` 逐项核对 3 个 skip 均为既有环境边界：TVM-available smoke 去重、
  2× frozen VNN-COMP checkout 不可用——与 B2-1 审计记录一致。
- black `--check` 5 文件 unchanged；mypy 4 文件 no issues；pylint `10.00/10`；
  `dol lint --soft` → `{"ok":true}`。
- 本轮纯 Python，无 C++/TIR vendored 改动，无需 rebuild（changelog 的
  "rebuild 通过/增量树无需重编译"声明与 diff 事实一致）。

### 7. claim 边界：PASS

- 五处措辞一致限定 sparse-source correctness：handoff、changelog、
  claims map（"不支持sparse-source融合…timing、speedup、memory或ASPLOS-ready"
  之外的本轮新增条目限定 correctness）、执行备忘录、current_status 均为
  `VALIDATED-B4-B2-B2-2-SPARSE-SOURCE-CORRECTNESS-PENDING-EXTERNAL-AUDIT`，
  `performance_claimed=false`、`sparse_source_admitted=true` 仅限 ABI correctness。
- 无 timing/fusion/性能外推；known-boundary 节明确 5 raw 是确定性重复而非性能证据。
- B2-3 开放范围（仅 P-anchor Conv dense correctness）与预注册 DAG 一致。

### 8. 上轮遗留处理：PASS（附 1 项 info 延续）

- changelog 76→77 更正已落实：`B2_1_DENSE_LINEAR_TIR_CHANGELOG` 行 42 现为
  "B4-B相关=`77 passed`（外审更正原76口径）"；预注册文档同步更正。
- B2-1 info"dense 侧缺 dtype/device/nonfinite 拒绝专项测试"：sparse 侧已补齐
  （`test_b4b2_sparse_linear_dtype_device_nonfinite_and_range_rejected`），
  但 dense 侧测试文件本轮未改动，仍无专项用例（dense runtime 校验本身存在，
  `fsg4_b4b2_dense_linear_tir.py` 行 270 isfinite/dtype/device 检查在）。
  记为延续 info，不阻塞。

## Findings

| severity | 位置 | 证据 | 建议 |
|---|---|---|---|
| info | `tests/test_fsg4_b4b2_dense_linear_tir.py` | dense 侧仍无 dtype/device/nonfinite 专项拒绝用例（10 个 test 无此项）；runtime 校验存在（runtime 行 263–270） | B2-3 或后续切片顺手补 dense 侧专项用例 |
| info | `boundflow/backends/tvm/differentiable_lower_sparse_linear.py` 行 369–375, 396–404 | forbidden workspace 检测为 scheduled script 子串计数，理论可被重命名绕过；但结构性保证（primfunc 无 dense buffer，审计方直读 script 确认仅 2 个 alloc_buffer）不依赖命名 | 可接受；若未来 schedule 复杂化，可考虑基于 buffer shape/alloc 语义的检测 |

## 不可现场复核项

- 无。runner、全量 pytest、静态检查、hash 重算、float64 独立重算均在本机
  RTX 4060 现场完成。

## claim-boundary 漂移判定

无漂移。全部文档与 receipt 把 claim 限定在 sparse-source correctness；
`performance_claimed=false` 在 Template/Schedule/Module/Launch 四层 receipt
均 fail-closed 强制。

## 下一动作判定

同意：B2-2 关闭，**仅开放 B2-3 P-anchor Conv dense correctness**。
timing、B2-4/B2-5、B4-B3 仍关闭。

## 附：关键命令

```bash
python scripts/run_fsg4_b4b2_sparse_linear_tir_correctness.py   # exit=0, 全部声明复现
python -m pytest -q tests/test_fsg4_b4b2_identity_tir.py tests/test_fsg4_b4b2_dense_linear_tir.py tests/test_fsg4_b4b2_sparse_linear_tir.py  # 34 passed
python -m pytest -q tests/test_fsg4_b4b*.py                     # 88 passed
python -m pytest -q -rs                                         # 1448 passed, 3 skipped
python -m black --check <5 files>; python -m mypy <4 files>; python -m pylint <5 files>  # 全过, 10.00/10
python3 /home/lee/Codes/codex-docops-logic/scripts/dol.py lint --soft  # ok
# 审计方独立 float64 闭合公式重算（无 autograd/无执行方 helper）：4 项输出 max diff ≤ 6.99e-07
```
