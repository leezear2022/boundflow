# FSG4/B4-B1 Typed Pure-PyTorch Reference 外部审计报告（Round 1）

- exchange：`fsg4-b4b1-typed-reference-20260818`
- 审计对象：`e62b387b9c92370db92c54f3c5b1e941574a4065` 及其 B4-B1 祖先提交
- request base：`88e0e7a`
- 审计方：external-model（独立外审）
- verdict：`request_changes`
- findings：blocker=0，major=2，minor=0，info=0

## 1. 结论

本轮不批准关闭 `VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE-PENDING-EXTERNAL-AUDIT`。
AC2、AC4、AC5、AC6 通过；AC1、AC3 因两个接口一致性缺口失败：receipt 未精确绑定预期
metric/gradient target 清单，execution context 未原样恢复 PyTorch deterministic debug/warn 模式。
两个缺口均已用独立最小复现直接触发，不能由 artifact 的 root replay 或现有回归测试替代。

B4-B2、CUDA/TIR、性能、显存、whole-core/query speedup 与 ASPLOS-ready 继续关闭。本轮
`request_changes` 不开放 B4-B2 预注册；只有 F1/F2 修复并经后续独立审计批准后，才可按原边界
另行预注册 B4-B2。

## 2. 审计身份与 provenance

- 审计绑定 code commit：`e62b387b9c92370db92c54f3c5b1e941574a4065`，而非审计时工作树
  HEAD `4bc413ef79727131270c992302cd2150b1f1e912`。
- source artifact：`artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1`。
- derived v2 artifact：`artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v2`。
- source manifest SHA-256：`67ace9e4a28c84306ee881a41aad1f16d9eddb7e80471ad76b030d691d9b25f6`。
- source protocol SHA-256：`a28d465274ff25ea704a21f376c45700f731983c0726e06aae3d13d424013406`。
- source summary SHA-256：`38db6fc1380630bdfaad815c44d64cb0a919a4f8c5c26cb85bd3070c91b42738`。
- derived v2 manifest SHA-256：`14923b0398f95b4adc0f95980d990a8f03260c173f0fd040f3ad5767e63f3167`。
- derived protocol SHA-256：`93e93cb15cbb7157330b257a204884e7a1c4649ec412575006c92048c8fd8394`。
- derived summary SHA-256：`becd8ae57536bc678392748bee5568d8b18922526df02da1238720b44045d744`。
- 正式 integrity report SHA-256：`6a3192f6a6ab2e14ab012bfedd3cc4251de416739ec6850290c98cd3aa399313`；
  probe code SHA-256：`154c90153d9fd2ac461957a1401c0d5184a59278dd239e956889fd95f731648b`。

独立核验了 225 个 raw payload tensor digest、10 条 base/reference capture hash 链以及
IR/instance/receipt/record hash 链；artifact 内记录的历史源码 digest 均与对应 `git show
<revision>:<path>` 内容一致。

## 3. AC1—AC6 判定

### AC1 — FAIL：Typed IR / Instance / Receipt 分层

通过项：

- 静态 IR 不携带 tensor payload 或单次 `base_capture_hash`。
- instance 绑定 base/reference capture 与完整 tensor digest。
- S/P 静态 IR hash 分别为
  `f5085dde03dde87310b90153f343717417f524b4892f4b0de210206007854a08`、
  `f781e56c8d10163031ee1e344e03b2599b9164be4de68316bb2f1838467f6f67`。
- 5 fresh 中 IR 各自唯一；instance/receipt 各 6 个唯一值，符合相同静态合同与不同动态实例的
  数据分布。
- lower-only、dense、single-consumer、default stream、no alias、shape/dtype/layout/presence
  等核心合同由 IR/instance 路径验证。

失败项见 F1：receipt 可在 `metrics=()` 且 `semantic_passed=True` 时通过验证，且
`incoming_lower_a_gradient_present` 可任意翻转而不被拒绝。因此 receipt 没有精确绑定预期
metric/result/gradient target inventory，AC1 的 fail-closed 要求不成立。

### AC2 — PASS：five-fresh raw 独立数值重算

审计脚本直接加载 5 个原始 PT，使用独立 torch eager 分解计算，不调用被审 reference helper，
结果为：

- runs=5，captures=10，metrics=60，elements=196,380；
- max absolute difference=`6.109476089477539e-07`；
- allclose=`true`，sign exact=`true`；
- S native alpha gradients=5/5，S native beta gradients=5/5；
- P native alpha gradients=5/5，P incoming-A gradients=5/5；
- P production beta shape=`(6, 0)`，beta gradient 缺席，不是 zero tensor；
- S production incoming-A 无 gradient target；5/5 forced-clone micro 结果与独立 eager 分解一致。

reference 仅使用公开 `torch` / `torch.nn.functional` 算子；未发现 TVM 或
`crown_ibp.py` 私有 oracle 导入。

### AC3 — FAIL：Deterministic v2 replay 与全局状态恢复

通过项：

- 从入口 torch threads=1/4/8 分别重算，三者 record JSONL SHA-256 均为
  `6042c4e1392feb3be9f1ca46ee8b64c29de26f4e3975c3b18e28b65342ab5687`。
- v2 root replay 逐字节通过，summary hash 为上述 `becd8ae5...d744`。
- v1 root replay 以 `ValueError: FSG4/B4-B1 reference protocol differs` 退出，未被静默接受。
- 普通 deterministic boolean、threads、precision、MKLDNN 状态在已覆盖情形可恢复。

失败项见 F2：调用方原先 `deterministic=True, warn_only=True, debug_mode=1` 时，context 退出后
变成 `deterministic=True, warn_only=False, debug_mode=2`，未原样恢复调用方全局状态。

### AC4 — PASS：negative integrity cases

正式两类 all-run、全链重签 negative integrity cases 已独立重跑，均先通过旧的 capture
sufficiency 层，再由 numerical reference 拒绝，结果 `2/2 rejected`：

1. all-run incoming lower bias；
2. all-run output lower-A adjoint。

另独立构造未注册用例：对全部 5 run 的 operator bias 加 `0.125`，同步更新 tensor digests、
reference capture hashes、PT digests、source summary/replay/manifest 与 derived protocol。该用例
完成全链重签后 source capture sufficiency 仍通过；变异 source manifest 为
`0822c2da99883cfb48864bbc8d0823bef4f95fc084102366e633e9b4cd5abe76`，变异 derived protocol
为 `87e8695e6a29ae68cba97852e2b8824245399ef5715228a372709a36c25117cb`，最终在 numerical
reference 的 `semantic-active-beta-gemm-14` 数值语义检查处拒绝。

### AC5 — PASS：回归、静态、diff 与 DocOps 门禁

审计时当前工作树的独立运行结果（不采信 delivery 声明数字）：

- targeted：`23 passed in 7.57s`；
- related：`119 passed, 12 skipped in 9.82s`；
- full：`1357 passed, 51 skipped, 7 warnings in 432.12s`；退出码均为 0。

本次数字与 request 中的旧声明值不同，是因为审计时仓库 HEAD/测试集合已有后续变化；因此仅将
本次命令的退出码和统计作为当前回归证据。51 个 skip 均由测试报告给出原因，主要为 CUDA 不可用、
冻结 VNN-COMP checkout 不可用及避免重复 TVM 编译；7 个 warning 为 torch JIT deprecation、
NVML 初始化与 treespec future warning。

静态/一致性结果：

- Black `--fast --check`：6 个 scoped 文件 unchanged；
- scoped Mypy `--explicit-package-bases`：5 个 source 文件无问题；
- Pylint：6 个 scoped 文件 `10.00/10`；
- `git diff --check 88e0e7a e62b387`：通过；
- `dol exchange validate` 与 `dol lint --soft`：提交前检查通过，提交后再次验证。

### AC6 — PASS：Claim 边界

独立检查 memo、claims map、current status、README、预注册、closure 及 artifact JSON/PT：共发现
38 个 `performance_claimed` / `tir_admitted` 字段，全部为 false。当前证据只支持 B4-B1 typed
reference correctness/gradient parity；v1 明确为 superseded。未发现 B4-B2/TIR、性能、显存、
whole-core/query speedup 或 ASPLOS-ready claim 漂移。

## 4. Findings

### F1 — major — Receipt 未精确绑定预期 metric 与 gradient target 清单

- 位置：`boundflow/runtime/fsg4_b4b1_pytorch_reference.py:516`，
  `DifferentiableLowerReferenceReceiptV1.validate`。
- 独立复现：从 run_00 分别构造合法 S/P receipt 后，将 receipt 替换为
  `metrics=(), semantic_passed=True`，两者 `validate(ir, instance)` 均通过；再单独翻转
  `incoming_lower_a_gradient_present`，两者也均通过。
- 精确输出：
  - `semantic-active-beta-gemm-14 original_metrics 6 empty_metrics_validated True incoming_flag False flipped_validated True`
  - `performance-conv-8-candidate original_metrics 6 empty_metrics_validated True incoming_flag True flipped_validated True`
- 原因：当前验证只要求 metric 名称唯一且排序，并用空集合的 `all(...) == True` 校验
  `semantic_passed`；没有从 IR/合同导出并强制完整 metric 名称集合，也没有验证
  `incoming_lower_a_gradient_present`。这使 typed receipt 可在没有任何数值证据或 target ownership
  错误时仍自称通过。
- 要求：从 IR/合同导出并精确验证 metric inventory（共同 5 项及 S native-beta / P incoming-A）；
  强制非空、名称/数量/target 完整匹配；使 beta/incoming flags 与 metric 及 production gradient
  presence 一致；加入直接 negative tests。

### F2 — major — Execution context 未原样恢复 deterministic debug/warn 模式

- 位置：`scripts/run_fsg4_b4b1_pytorch_reference_artifact.py:263`，
  `_reference_execution_policy`。
- 独立复现：进入 context 前调用
  `torch.use_deterministic_algorithms(True, warn_only=True)`；进入前状态为
  `True True 1`，context 内为 `True False 2`，退出后仍为 `True False 2`。
- 精确输出：
  - `before True True 1`
  - `inside True False 2`
  - `after True False 2`
- 原因：context 只保存 boolean `are_deterministic_algorithms_enabled()`，恢复时调用
  `torch.use_deterministic_algorithms(previous_deterministic)`，丢失原 `warn_only`/debug mode。
  这会把调用方后续的 deterministic warning 改成 hard error。
- 要求：保存并恢复 `torch.get_deterministic_debug_mode()`（或等价 warn-only 状态），覆盖 mode
  0/1/2、正常退出和异常退出测试，并继续校验 threads/precision/MKLDNN 等全部 policy 字段。

## 5. 不可现场复核项与边界

不可现场复核项：无。审计直接重放冻结的 5 个原始 PT；未重新生成 GPU captures，因为本轮验收对象
是已绑定 provenance 的冻结 raw artifact，且当前环境无 CUDA。该限制不影响 raw 数值、hash 链、
deterministic replay、negative integrity cases 或两个 findings 的独立复现。

## 6. 正式处置

- verdict：`request_changes`；
- 不同意 executor 关闭本 exchange；
- F1、F2 修复后应重新 delivery 并接受下一轮外审；
- 在后续审计批准前，B4-B2、CUDA/TIR 与全部性能类 claims 继续关闭。
