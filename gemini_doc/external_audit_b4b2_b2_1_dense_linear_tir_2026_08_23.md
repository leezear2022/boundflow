---
status: external-audit-complete
updated: 2026-08-23T05:40:00Z
type: external-audit
topic: boundflow
slug: fsg4-b4b2-b2-1-dense-linear-external-audit
stage: s01
---

# FSG4/B4-B2 B2-1 Dense Linear TIR 独立外部审计报告

- 审计对象：branch=`feat/rvir-v4-production-state-ownership-v1`，source=`eb74e45`，
  handoff=`2da99da`(HEAD==origin，已核实),base=`09c559d`,preregistration=`57be636`。
- 审计环境：本机 RTX 4060 Laptop GPU(sm_89),conda env `boundflow`,Torch 2.12.1+cu132。
- 审计方式：不信任交接数字，全部现场复跑/独立重算/亲读代码。

## 总体 Verdict

**approve**(0 blocker / 0 major;1 minor、3 info，均不构成本切片门禁问题)。

B2-1 的实现、实测证据与预注册门禁逐条吻合；数学语义经独立 float64 重算确认；
claim 边界无漂移；B2-0 遗留两条 minor 均已关闭。**同意开放 B2-2
S-anchor sparse-source fused forward/backward。**

## 逐项结论(对照复核清单 8 项)

### 1. git/范围 —— PASS

- `git log`:`57be636`(预注册)→`712ca03`(B2-0)→`09c559d`(B2-0 外审关闭)→
  `eb74e45`(B2-1 实现)→`2da99da`(交接文档)。B2-0 外审批准先于 B2-1 代码，顺序正确。
- `git show --stat eb74e45`:18 文件，+2403/-8。新增 5 个代码文件(IR 637 行、backend 335 行、
  runtime 768 行、runner 117 行、测试 356 行)+ 文档；修改仅
  `boundflow/runtime/fsg4_b4b2_identity_tir.py`(13 行)与
  `tests/test_fsg4_b4b2_identity_tir.py`(+12 行)。
- B2-0 文件 13 行改动逐行审查:`_IdentityTIRExecutor` 新增 `fallback_count`/
  `eager_backward_count` 计数器与 `reject_fallback()`(先计数再 raise),receipt 由硬编码
  `0` 改为读真实计数器。**只增不改语义**,正是上轮 minor #1 的关闭方式。
- `git diff 09c559d eb74e45` 无 B2-2 sparse/fused、Conv、timing、same-solver、
  optimizer 或旧 production 执行路径改动。
- 工作树仅 `M .docops/ev.jsonl`(DocOps hook 自动事件)。

### 2. 预注册一致性 —— PASS

- 预注册 §6 B2-1 定义:"实现 Linear/Gemm dense semantic ABI。5 个 B4-B1 raw instances 逐项比较
  forward A/bias、incoming-A clone、native α、active native β gradients;先用确定性 correctness
  schedule,不计时"。实现逐条对应:5 raw × 4 metric(forward A/bias、α/β gradient),incoming-A
  以执行前后 hash 不变门禁覆盖;schedule=`dense-linear-serial-reduction-v1` 且
  `performance_admitted=false`。
- 容差 atol=rtol=2e-4(§7)、5 fresh、S-anchor shape `[6,1,100]@[100,1024]`、sm_89、
  float32/cuda 全部一致。
- `git show eb74e45` 对预注册文档的 11 行改动仅为 status 头与"内部结果"注解,
  正文门禁(B2-1 定义、§7 容差、§10 claim ledger)未动。**无事后挪门禁。**

### 3. 数学正确性 —— PASS(重点亲验)

亲读 TIR(`boundflow/backends/tvm/differentiable_lower_dense_linear.py`)并对预注册 §2
合同逐条推导:

- forward:`upper_slope`(l≥0→1;u≤0→0;否则 u/max(u−l,ε),ε=float32 eps,与 reference
  `clamp_min(eps)` 完全一致)、`lower_slope`(ambiguous→clamp(α,0,1))、
  `selected_slope/intercept`(A≥0 选 lower)、`R = A·slope − β·split`(active β signed
  pre-add)、`Y = R @ W`(weight `[100,1024]` 自然布局,`W[c,p]` 收缩维 c)、
  `output_bias = incoming_bias + Σ_c(A·intercept + R·op_bias)` —— 与合同逐条一致。
- backward:`adjoint_matmul = dY @ Wᵀ`(以 `W[c,p]` 索引实现转置收缩)、
  `adjoint_relu = adjoint_matmul + dB·op_bias`、`dα = Σ_s adjoint·A·1{A≥0, l<0<u, 0≤α≤1}`
  (clamp 端点含 0/1,与 PyTorch `clamp` 导数选择一致;A==0 时因乘 A 而为 0)、
  `dβ = Σ_s −adjoint·split`(`+ β·0` 仅为保活 buffer,数值恰为 0)。与合同的离散导数
  所有权逐条一致。
- reduction 语义:schedule 只 bind 空间维到 blockIdx/threadIdx(128 threads),reduction 轴
  保持串行,每输出元素单线程顺序累加 → deterministic;schedule IR 显式声明
  `deterministic=true`、`dense-linear-serial-reduction-v1`。
- **reference 独立性**:对比方是 `run_b4b1_pytorch_reference_v1`(B4-B1 纯 PyTorch eager
  oracle,B4-B1 已经 Round-2 独立外审批准),非同一 TIR 自比。
- 此外我用自写 PyTorch 表达式(仅依预注册合同推导,**float64 精度**,不复用 repo
  reference 代码)对 5 份 raw 全部 4 项输出独立重算:36,750 元素,全部
  allclose(2e-4)/sign exact/finite,**对 float64 ground truth 的最大差 =
  6.988e-07**(output_lower_a 4.268e-08、output_bias 5.128e-08、α grad 6.694e-07、
  β grad 6.988e-07),与声明的 8.64e-07(对 float32 reference)同量级且方向一致。
- S-anchor 梯度所有权:β gradient 每 run 恰 6 个非零位置(TIR 与 reference 一致);
  incoming A production requires_grad=false,reference 亦返回 None → incoming-A gradient
  absent 符合合同;P-anchor(captures[1],requires_grad=true)被 admission 拒绝(专项测试)。

### 4. 实测独立复核 —— PASS(逐位一致)

现场重跑 `python scripts/run_fsg4_b4b2_dense_linear_tir_correctness.py`(fresh process、GPU):

| 声明 | 交接值 | 审计方现场值 | 结果 |
|---|---|---|---|
| run 数 | 5 | 5 | 一致 |
| metric 数 | 20 | 20 | 一致 |
| 元素数 | 36,750 | 36,750(=5×(6144+6+600+600)) | 一致 |
| max abs diff | 8.642673492431641e-07 | 8.642673492431641e-07 | **逐位一致** |
| allclose/sign | true/true | true/true | 一致 |
| template hash | d96bb8d6…0a0be4 | d96bb8d6…0a0be4 | **逐位一致** |
| schedule hash | 989c3eae…a5de4b | 989c3eae…a5de4b | **逐位一致** |
| module receipt | e9912143…d80801a | e9912143…d80801a | **逐位一致** |
| cache | miss,hit,hit,hit,hit | miss,hit,hit,hit,hit | 一致 |
| launch(fwd/bwd) | 每 run 1/1 | (1,1)×5 | 一致 |
| fallback/eager | 0/0 | 0/0×5 | 一致 |

- 23 个 DLPack pointer(9+2 forward、10+2 backward)exact、stream 双向一致、输出不 alias
  输入由 receipt validate 强制(`launch.validate_against` 不过即 raise)。
- 5 份 raw 文件 sha256 互不相同，但 S-anchor capture 数值相同(5 run 的 4 项 metric hash
  完全一致)——交接 §5 已披露"确定性重复 capture，不等于过程性能证据",本阶段不计时，
  记为 info。

### 5. 测试质量 —— PASS

`tests/test_fsg4_b4b2_dense_linear_tir.py`(356 行，10 个 test,无 parametrize 注水):

- 覆盖:IR round-trip + fail-closed(ABI/schedule/performance_admitted 篡改)、P-anchor 拒绝、
  five-fresh parity(具体断言 allclose/sign/cache/launch/fallback/β非零=6/module hash 唯一)、
  receipt round-trip + 23 pointer、clamp 端点(α=0/1)梯度、A=0 离散导数(α VJP=0)、
  custom-stream 异常后 device/stream/deterministic policy 恢复、真实 fallback 计数器、
  higher-order 拒绝、重签 receipt(instance ordinal、tvm_commit、fallback_count、
  performance_claimed)拒绝。
- 断言具体(数值容差、计数器、hash、非零计数),无 trivially-true 测试。
- 现场运行:23 passed(identity 13 + dense 10)。
- info:dense 侧无显式的 dtype/device/nonfinite 拒绝专项测试(validation 代码存在于
  `_validate_dense_linear_tensors`,identity 侧有同类测试),建议 B2-2 补齐。

### 6. 测试/静态 —— PASS

- targeted:`pytest -q tests/test_fsg4_b4b2_identity_tir.py tests/test_fsg4_b4b2_dense_linear_tir.py`
  → **23 passed**。
- B4-B 相关:`pytest -q tests/test_fsg4_b4b*.py` → **77 passed**(交接声称 76;差异 +1,
  原因是 eb74e45 给 identity 测试新增了 1 个 fallback 计数器拒绝测试——执行方统计口径
  先于该测试合入。方向安全，记为 minor 报告精度问题)。
- 全量:`pytest -q -rs` → **1437 passed, 3 skipped, 6 warnings in 456.27s**。3 个 skip 逐项核对:
  `test_artifact_phase5d_smoke.py:118`(TVM 可用时跳过 allow-no-tvm smoke)、
  `test_cross_axis_verification_batch_artifacts.py:70` 与
  `test_root_projection_floor_artifacts.py:66`(frozen VNN-COMP checkout 不可用)——均为既有
  环境边界，与 B2-1 无关。
- `black --check`(7 文件):would be left unchanged;`mypy`(5 文件):no issues;
  `pylint`(7 文件):10.00/10;`bash scripts/rebuild_tvm.sh`:Rebuild Complete(增量无需
  重编译);`dol lint --soft`:ok=true。

### 7. claim 边界 —— PASS,无漂移

- IR 强制:`enabled_by_default=false`、`performance_admitted=false`(template/schedule)、
  `performance_claimed=false`(module/launch receipt),篡改即 fail closed(有测试)。
- runner 输出 `performance_claimed=false`、`sparse_source_admitted=false`;新代码无任何
  timing API(grep `perf_counter|cuda.Event|elapsed` 无命中)。
- claims map/备忘录/master plan/README 新增条目均严格限定
  "dense Linear correctness pending external audit",明确排除 sparse 融合、timing、speedup、
  memory、ASPLOS-ready。`.docops/s.md` next=仅外审。
- dense workspace(`output_bias_delta`/`adjoint_matmul`/`adjoint_relu`)在 TIR 与 schedule IR
  中显式保留，未藏入任何计时区(本切片无计时区)。
- **B2-0 两条 minor 处理结论**:
  - minor #1(fallback_count 硬编码):**已关闭**。identity 与 dense executor 均为真实计数器,
    `reject_fallback` 先计数再 raise，两侧各有专项拒绝测试。
  - minor #2(rebuild 措辞):**已关闭**。B2-1 changelog 改为"完成且增量树无需重编译"，
    不再逐字引用 ninja 行;审计方现场 `rebuild_tvm.sh` 输出 "Rebuild Complete!"。

### 8. B2-2 门禁评价 —— 一致

交接"approve 且无 blocker/major 才开放 B2-2 S-anchor sparse-source fused forward/backward"
与预注册 §6 DAG(B2-0→B2-1→B2-2)及 §10 claim ledger 一致;B2-2 失败即
`VALIDATED-NO-GO-B4-B2-SEMANTICS` 的 fail-closed 语义未被削弱。本审计结论为 approve 且
0 blocker/0 major,**同意开放 B2-2**。timing、P-anchor、B2-4/B2-5、B4-B3 仍关闭。

## Findings

| severity | 位置 | 证据 | 建议 |
|---|---|---|---|
| minor | changelog/handoff "B4-B相关=76 passed" | 现场 `pytest -q tests/test_fsg4_b4b*.py` = **77 passed**;差异为 eb74e45 新增的 identity 计数器测试 | 后续文档按 77 更正;不影响结论 |
| info | 5 份 raw 为确定性重复 capture | 5 run 的 4 项 metric hash 完全一致;run_0*.pt 文件 sha256 互不相同 | 交接 §5 已披露;B2-5 formal artifact 阶段保留 raw stdout 并按预注册 §9 绑定 |
| info | dense 测试无显式 dtype/device/nonfinite 拒绝用例 | `_validate_dense_linear_tensors` 有校验,identity 测试有同类覆盖 | B2-2 一并补齐 |
| info | TIR `max(u−l, ε)` 与 reference `clamp_min(ε)` | 两侧完全一致(float32 eps),非合同偏离 | 无需处理 |

## 不可现场复核项

- 执行方当次运行的原始 stdout(未冻结 artifact)——已由审计方现场重跑替代，三项 receipt
  hash 与全部数值逐位一致，风险消除。
- "5 fresh process"的进程隔离性:交接以单 process 内 5 次 fresh capture 重放实现;数值确定性
  已由审计方独立 process 重跑 + float64 独立重算双重确认。

## 附录:关键命令与输出摘录

```bash
# 现场 runner(逐位一致)
$ python scripts/run_fsg4_b4b2_dense_linear_tir_correctness.py
{"allclose":true,"compute_capability":"sm_89","device":"NVIDIA GeForce RTX 4060 Laptop GPU",
 "element_count":36750,"maximum_absolute_difference":8.642673492431641e-07,"metric_count":20,
 "module_receipt_hash":"e99121435e5db022c02f1d1610ffb9d4048397e09168f91f6857e425ad80801a",
 "performance_claimed":false,"sparse_source_admitted":false, ... cache: miss,hit,hit,hit,hit,
 template_hash: d96bb8d62eb2e112e4f9ac5e98bc971cb41122cd97273ebb3fc1c4fc5c0a0be4,
 schedule_hash: 989c3eae7fcefed3a6399b000c51eb222c5e5ba2a31a220ef42db5d86ca5de4b}

# 审计方独立 float64 重算(.tmp/audit_b2_1_independent_recompute.py)
TOTAL elements=36750 global_max_vs_f64=6.987927e-07 OK=True
# 每 run: output_lower_a 4.268e-08 / output_bias 5.128e-08 / α grad 6.694e-07 /
#         β grad 6.988e-07;beta_nonzero=6;clamp_endpoint_sites=78;allclose/sign/finite 全 true

# 测试与静态
23 passed (targeted) / 77 passed (tests/test_fsg4_b4b*.py) / 1437 passed, 3 skipped (full)
black: 7 files unchanged; mypy: no issues in 5 files; pylint: 10.00/10
rebuild_tvm.sh: Rebuild Complete!; dol lint --soft: {"ok":true}
```
