# RVIR 审计后加固外部复审交接

> 日期：2026-08-03
> 审计基线：`b01225b3ffa7a4bfc6fcdfe42ab9fe0973d631ae`（原 RVIR 审计内容合入 main）
> 待审结果：`main@1a6eb65`（PR #5—#8 全部合并并完成 DocOps 同步）
> DocOps task：`rvir-post-hardening-20260803`

## 1. 复审目标

请不要采信本文给出的测试数字，独立核对原审计的五个 minor finding 在 PR #5—#8 后的
真实状态，并检查这些修复是否引入 claim drift 或回归。

原审计结论仍是 RVIR CPU correctness/integration 的 **VALIDATED-REDUCED**。本轮复审只判断
审计后加固是否闭环，不得把它升级为 performance、CUDA、fused-kernel、完整 verifier E2E
或 ASPLOS-ready 结论。

## 2. 待审提交范围

| 原 finding | 后续 PR / merge | 主要变化 |
|---|---|---|
| F1 / M1 | PR #5 / `7437099` | 固化 backend mismatch、missing emit、duplicate launch、local semantics owner 四条拒绝路径 |
| F2 / M2 | PR #6 / `7358407` | Phase6H 显式选择 Python；受限 PATH 可运行；缺失解释器 fail closed |
| F3 / M3 | protocol | 原 `rvir-20260803/r001` 保持不改；本轮使用全新 DocOps task，禁止覆盖旧 round |
| F4 / M4 | PR #7 / `6a41439` | v2 artifact 冻结 377 条原始 query 与 377 条 typed record，并进行语义 replay |
| F5 / M5 | PR #8 / `5b54856` | 固定 upstream/input 的 ResNet 原始数值连续重跑及 tensor digest 证据 |

审计范围使用：

```bash
git diff --stat b01225b3ffa7a4bfc6fcdfe42ab9fe0973d631ae..1a6eb65
git log --oneline b01225b3ffa7a4bfc6fcdfe42ab9fe0973d631ae..1a6eb65
```

## 3. Acceptance criteria

### AC1：RVIR-2 拒绝路径已固化

- 四条专用负向测试实际存在且不是只断言任意异常；
- backend identity 不匹配必须在 exact call 前拒绝；
- missing emit、duplicate launch 必须由 schedule validation 拒绝；
- `semantics_owner=boundflow` 必须被 external-call ownership 门禁拒绝；
- 正常 external exact call 仍只 launch/emit 一次。

### AC2：Phase6H Python 路由不再依赖调用者 PATH

- `PHASE6H_PYTHON` 明确指定当前 conda Python 时，即使 `PATH=/usr/bin:/bin` 也能生成完整
  artifact；
- 未设置 override 时，激活的 `CONDA_PREFIX/bin/python` 可被选择；
- 指定不存在的解释器时退出码为 2，且在产生 artifact 前 fail closed；
- 不得把 `/usr/bin/python` 缺少 torch 的环境错误掩盖成测试通过。

### AC3：DocOps exchange 不可变性保持

- 原任务 `.docops/exchange/rvir-20260803/r001/` 的 audit/delivery 内容与关闭状态保持有效；
- 新复审使用 `.docops/exchange/rvir-post-hardening-20260803/`，没有覆盖旧 task/round；
- `dol exchange validate` 与 `dol lint --soft` 均通过。

### AC4：v2 在线原始证据可独立语义重放

- artifact 内确有 377 条 query 与 377 条 typed record，ID/sequence 一一对应；
- root/parent = 30/347，parent 必须先于 child；377 条全部 completed；
- replay 对 377 条 query 重新编译五层 IR，并逐行比较 Bound、PlanTemplate、PlanInstance、
  Task、Schedule hash；
- digest 全部先验校验；即使攻击者重写 payload 并更新 manifest digest，伪造 schedule hash 或
  parent 顺序仍必须被语义门禁拒绝；
- v1 artifact 保持不可变且继续可 replay；历史 fused coverage 仍为 0/394。

### AC5：ResNet 原始数值可在固定 external 环境重跑

- αβ-CROWN commit、auto_LiRPA submodule、VNN-COMP commit、ONNX SHA256、vnnlib SHA256
  必须全部与冻结身份一致；
- fresh CPU run 的 manifest `status=ok`；
- 12 个冻结语义字段逐项相等，包括 6 组 external intermediate bounds、adaptive slope、
  intermediate hash、max diff 与 sign；
- 原始 tensor digest 与记录值一致；
- 重跑只用于 correctness，不得引用单次 timing 形成性能结论。

### AC6：总体回归和 claim boundary

- 合并后 main 的 targeted tests、full suite、Black、mypy、Pylint 均通过；
- CUDA skip 必须被报告，不能算作 CUDA 验证；
- RVIR 仍为 VALIDATED-REDUCED，IR-5 仍为 VALIDATED-NO-GO；
- 不得出现 performance、GPU、BoundFlow 本地 αβ kernel、完整 VNN-COMP E2E 或
  ASPLOS-ready 的主张漂移。

## 4. 建议独立复核命令

### 4.1 合并后测试与 replay

```bash
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate boundflow

python -m pytest -q \
  tests/test_real_verifier_ir_integration.py \
  tests/test_phase6h_artifact_runner_smoke.py \
  tests/test_real_verifier_ir_artifact.py

python scripts/run_real_verifier_ir_artifact.py replay \
  --artifact-dir artifacts/rvir/rvir-cpu-correctness-v1-20260803

python scripts/run_real_verifier_ir_artifact.py replay \
  --artifact-dir artifacts/rvir/rvir-cpu-correctness-v2-20260803

python -m pytest -q tests
```

不要只运行正向 replay。请阅读并运行
`tests/test_real_verifier_ir_artifact.py` 中重写 payload 与 manifest digest 后的 tamper cases，
确认其失败来自语义重算而不是旧 digest。

### 4.2 静态门禁

```bash
python -m black --check \
  scripts/run_real_verifier_ir_artifact.py \
  tests/test_real_verifier_ir_artifact.py \
  tests/test_real_verifier_ir_integration.py \
  tests/test_phase6h_artifact_runner_smoke.py

python -m mypy \
  scripts/run_real_verifier_ir_artifact.py \
  boundflow/runtime/verifier_ir_integration.py

python -m pylint \
  scripts/run_real_verifier_ir_artifact.py \
  tests/test_real_verifier_ir_artifact.py \
  tests/test_real_verifier_ir_integration.py \
  tests/test_phase6h_artifact_runner_smoke.py
```

### 4.3 F5/M5 external 环境复现

使用独立临时目录，不修改 BoundFlow vendored submodule：

```bash
AUDIT_ROOT="$(mktemp -d)"
ABCROWN_ROOT="$AUDIT_ROOT/alpha-beta-CROWN"
VNNCOMP_ROOT="$AUDIT_ROOT/vnncomp2021"
RERUN_OUT="$AUDIT_ROOT/resnet-rerun"

git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git "$ABCROWN_ROOT"
git -C "$ABCROWN_ROOT" checkout e5c7e17bf0488843acb77b7519f59876717a49f4
git -C "$ABCROWN_ROOT" submodule update --init auto_LiRPA
test "$(git -C "$ABCROWN_ROOT/auto_LiRPA" rev-parse HEAD)" = \
  5a098e8f9fb5786a428a024981d833d303921f2d

git clone --filter=blob:none --no-checkout \
  https://github.com/VNN-COMP/vnncomp2021.git "$VNNCOMP_ROOT"
git -C "$VNNCOMP_ROOT" sparse-checkout set \
  benchmarks/cifar10_resnet/onnx \
  benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered
git -C "$VNNCOMP_ROOT" checkout 90419aadcf06cf543ce5c1706cae1059dc9fa6cf

sha256sum \
  "$VNNCOMP_ROOT/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx" \
  "$VNNCOMP_ROOT/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"

python scripts/replay_pr14_abcrown_initial_crown.py \
  --abcrown-root "$ABCROWN_ROOT" \
  --model "$VNNCOMP_ROOT/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx" \
  --vnnlib "$VNNCOMP_ROOT/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib" \
  --output-dir "$RERUN_OUT" \
  --workload-name vnncomp21-resnet2b-prop0-cpu-rvir1 \
  --device cpu --warmup 0 --repeats 1 --backends pytorch_eager
```

然后独立读取 `$RERUN_OUT/manifest.json` 与 `payload.pt`，不要复用交接文档中的比较结果。
输入和 tensor 的预期 digest 记录在
`gemini_doc/change_2026-08-03_rvir_resnet_raw_rerun.md`。

### 4.4 DocOps

```bash
python3 /path/to/docops/scripts/dol.py exchange validate rvir-20260803
python3 /path/to/docops/scripts/dol.py exchange validate rvir-post-hardening-20260803
python3 /path/to/docops/scripts/dol.py lint --soft
```

## 5. 当前执行方结果（必须独立复核）

- targeted：`12 passed in 31.99s`；
- v2 fresh replay：`activation_call_count=394`，`status=replayed`；
- full suite：`460 passed, 37 skipped, 5 warnings in 76.71s`；
- Black：4 files unchanged；mypy：2 files clean；Pylint：10.00/10；
- 第三次 fresh ResNet run：`status=ok`、冻结字段匹配、记录 tensor digest 匹配；
- ResNet max diff `3.0994415283203125e-06`、sign 9/9、intermediate hash
  `d51615b04dfb205afd67d2c21680ece4ca92f693157da1e32c7f8202a8e08cf1`；
- 旧 exchange validation：PASS。

## 6. 不得删除或升级的限制

1. fused replacement coverage 历史事实仍为 `0/394`；
2. 394/394 是 typed external exact-call admission，不是 BoundFlow fused kernel execution；
3. 377/377 是 simple MLP CPU online exact-call execution；
4. ResNet 只覆盖 initial plain-CROWN external intermediate semantics；
5. 当前没有 fresh CUDA evidence；
6. lower-only external 与 lower+upper BoundFlow 的性能公平合同仍未建立；
7. 没有完整 verifier E2E 或性能结论；
8. IR-5 保持 VALIDATED-NO-GO，IR-6 不启动，ASPLOS-ready 仍为 NO。

## 7. 审计报告格式

请在新 DocOps task 的当前 round 返回：

1. 总体 verdict：`approve` 或 `changes_requested`；
2. AC1—AC6 逐项结论与独立命令证据；
3. F1—F5 当前状态：closed / partial / open；
4. blocker / major / minor finding，包含文件和可操作建议；
5. 不可现场复核项及原因；
6. claim-boundary 是否有漂移；
7. 最终建议的下一工程动作。
