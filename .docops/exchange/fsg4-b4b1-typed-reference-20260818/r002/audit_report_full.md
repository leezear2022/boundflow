# FSG4/B4-B1 Typed Pure-PyTorch Reference 外部审计报告（Round 2）

- exchange：`fsg4-b4b1-typed-reference-20260818`
- round：2
- request base：`88e0e7a`
- Round 2 clean source：`e711e991bed54a16c881a2f2bbeb18d71de3c210`
- v3 artifact commit：`b8213e2e31a991df77ad484dfaa760b969fe9bd1`
- delivery result commit：`80d6ca4`
- 审计方：external-model（独立外审）
- verdict：`approve`
- findings：blocker=0，major=0，minor=0，info=0

## 1. 结论

Round 1 的 F1、F2 均已关闭，AC1—AC6 全部通过。本轮批准 executor 关闭
`fsg4-b4b1-typed-reference-20260818` exchange，并关闭
`VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE-PENDING-EXTERNAL-AUDIT`。

批准后只允许另行预注册 B4-B2；本轮不代表 B4-B2/CUDA/TIR 已实现，也不支持性能、显存、
whole-core/query speedup 或 ASPLOS-ready claim。

## 2. 审计方法与提交身份

审计先执行正式 `audit-start`，完整读取 request、Round 1 audit、F1/F2 immutable responses、
Round 2 delivery 与 Round 2 handoff。未采信 delivery 中的汇总数字；所有数值、直接负例、replay、
回归和静态门禁均由本轮本地命令独立取得。

delivery 的 `80d6ca4` 是内部关闭/交接文档提交；实际修复源码来自 `e711e99`，v3 artifact 来自
`b8213e2`。审计按这两个身份核对历史源码与 artifact，而不是把审计时 HEAD
`5252701dd5c1c67bb9f857105034a2ea9cf0c2f9` 当作 artifact source。

## 3. Round 1 findings 关闭复核

### F1 — CLOSED：exact receipt inventory / target binding

`DifferentiableLowerReferenceReceiptV1.validate()` 现从 IR contracts 导出精确 target map：共同五项
`native_alpha`、`native_alpha_gradient`、`native_beta`、`output_bias`、`output_lower_a`；S 追加
`native_beta_gradient`；P 追加 `incoming_lower_a_gradient`。它同时强制：

- metric 名称集合、数量与排序精确匹配；
- `element_count == prod(contract.shape)`；
- `production_hash == instance.input_tensor_hash_map[target_contract]`；
- beta/incoming presence flags 与 IR gradient ownership 一致；
- reference/production digest 为 64 位小写 SHA-256；
- `semantic_passed` 由完整 metrics 的 allclose/sign exact 导出。

独立从 `run_00.pt` 构造合法 S/P receipt 后，对两个 anchor 分别执行以下 10 类直接 negative
cases：空 metrics、删一项、翻转 incoming flag、翻转 beta flag、替换 production hash、替换
element count、打乱 metric 顺序、将 reference digest 改为大写、把 semantic_passed 改为 false、
交换两个 production target hashes。共 20/20 均以 `ValueError` fail closed，未依赖 artifact 外层
digest。

独立观察的合法 inventory：

- S：`native_alpha(600)`、`native_alpha_gradient(600)`、`native_beta(600)`、
  `native_beta_gradient(600)`、`output_bias(6)`、`output_lower_a(6144)`；flags=`beta true / incoming false`。
- P：`incoming_lower_a_gradient(6144)`、`native_alpha(6144)`、
  `native_alpha_gradient(6144)`、`native_beta(6144)`、`output_bias(6)`、
  `output_lower_a(6144)`；flags=`beta false / incoming true`。

每项 production hash 均与对应 instance target digest 相等。F1 关闭。

### F2 — CLOSED：deterministic debug/warn exact restoration

独立把调用方 deterministic debug mode 依次设为 0/1/2，并分别覆盖正常退出和 context 内抛出
异常，共 6 组。每组 context 内均为：

`threads=1, debug_mode=2, warn_only=false, precision=highest, MKLDNN=false`

退出后均逐项恢复：`threads=4`、原 debug/warn mode、`precision=medium`、`MKLDNN=true`。尤其
mode=1 的两组均从 warn-only 恢复为 warn-only，未升级为 hard error。正常与异常路径结果相同，
F2 关闭。

## 4. AC1—AC6 判定

### AC1 — PASS：Typed IR / Instance / Receipt fail closed

除 F1 的独立负例矩阵外，静态 IR 仍不携带 tensor payload 或单次 base capture hash；instance
绑定完整 tensor inventory；lower-only、dense、single-consumer、default stream、no alias 及
shape/dtype/layout/presence 合同未放宽。S/P 静态 IR hash 保持：

- S：`f5085dde03dde87310b90153f343717417f524b4892f4b0de210206007854a08`
- P：`f781e56c8d10163031ee1e344e03b2599b9164be4de68316bb2f1838467f6f67`

### AC2 — PASS：five-fresh raw 独立数值重算

独立脚本直接加载 5 个 PT，使用 raw tensors 与公开 torch eager 算子重建 sparse alpha/beta、
lower sign-select、intercept reduction、Linear/Conv transpose contraction及 local VJP；未调用被审
reference helper。按 v3 protocol 独立固定 CPU policy 后结果：

- runs=5，captures=10（S/P 各 5）；
- metrics=60，elements=196,380；
- maximum absolute difference=`6.109476089477539e-07`；
- allclose=`true`，sign exact=`true`；
- S native alpha/beta gradients=5/5；
- P native alpha/incoming-A gradients=5/5；
- P production beta shape=`(6, 0)` 且 beta gradient absent=5/5；
- S production incoming-A gradient absent=5/5。

作为环境敏感性对照，同一独立 eager 代码在未冻结默认 CPU policy 下得到 max diff
`1.9073486328125e-06`，仍 allclose/sign exact；按 protocol 冻结后精确回到正式值，说明执行策略
冻结有实际作用，而非只存在于 metadata。

### AC3 — PASS：v3 protocol、state restoration 与 fail-closed replay

protocol 精确冻结并由 validator 强制：

- `torch_num_threads=1`
- `torch_deterministic_algorithms=true`
- `torch_deterministic_debug_mode=2`
- `torch_deterministic_state_restore=exact-debug-mode-v1`
- `torch_float32_matmul_precision=highest`
- `torch_mkldnn_enabled=false`
- `receipt_metric_inventory=exact-ir-contract-target-v1`

独立把 debug mode、restore policy、receipt inventory 三个字段分别改写并重算 protocol hash，三类
均由 `_validate_protocol` 拒绝。v1、v2 root replay 均退出 1，拒绝点为
`FSG4/B4-B1 reference protocol differs`；v3 root replay 退出 0、逐字节通过，records JSONL hash
为 `6042c4e1392feb3be9f1ca46ee8b64c29de26f4e3975c3b18e28b65342ab5687`。

### AC4 — PASS：v3 provenance 与 negative integrity cases

独立以 Python 标准库重算 canonical hashes、全部 artifact file digests、5 个 raw PT digests，并用
`git show e711e99:<path>` 重算 6 个历史源码 digest，全部匹配。冻结 identity 为：

- manifest：`2f8a1ffde0f99777e0ab6d9dddb1042c2f7f6c71e57882d141035553475e4e3f`
- protocol：`b95bc20c8dcaef8635741842b85d4d0bf9e41c9592c60896677907cd96914baf`
- summary：`753a9558a7c36cb89f02963dcd08fc8e76fdfcd415f7dc5d969eea77dffc7a0b`
- integrity report：`50a12f577d60a8bf115ee8c40b248f88ecd451715a4b0b4a2f420dedc4aec964`

正式 probe 独立重跑得到 `2/2 rejected`：all-run incoming-bias 与 all-run output-adjoint 均完成
内部 capture、source summary/manifest 与 derived protocol 重签，最终由 numerical reference 在
S anchor 拒绝。运行时报告因 `source_git_head` 为当前审计 HEAD，整体 report hash 与正式冻结值不同，
但 case 结果、probe digest、reference code digests 与拒绝点一致。

未注册直接负例包括 S/P production target hashes 互换、uppercase reference digest、metric 顺序
变更，以及重签 protocol 后的 policy/inventory 字段变更；均在对应 receipt/protocol 语义层拒绝。

### AC5 — PASS：回归、静态、diff 与 DocOps

本轮独立结果：

- targeted：`32 passed in 7.23s`
- B3/B4 related：`128 passed, 12 skipped in 10.11s`
- full：`1366 passed, 51 skipped, 7 warnings in 421.08s`
- Black：3 files unchanged
- scoped Mypy：2 source files，no issues
- scoped Pylint：`10.00/10`（只出现只读 home cache 无法写 stats 的非代码 warning）
- `git diff --check e62b387..b8213e2`：通过
- `dol exchange validate`、`dol lint --soft`：提交前通过，提交后再次验证

当前审计进程使用 `torch 2.12.1+cu132`，但 `torch.cuda.is_available()=false`、device_count=0，
`nvidia-smi` 无法连接 driver，因此不能冒充 RTX 4060 现场 GPU 复跑。全量 51 个 skip 可精确分解为
48 个 CUDA 条件 skip，加 3 个非 CUDA skip；后者正是 allow-no-TVM 重复编译与两个冻结
VNN-COMP checkout 不可用。总收集数为 1417，若 48 个 CUDA 条件项在 RTX 4060 上执行，则边界为
1414 passed / 3 skipped，与 delivery 声明的集合边界一致。B4-B1 targeted 为 32/32，无 skip；
本轮不把 GPU 条件项的 executor 结果当成独立性能证据。

### AC6 — PASS：Claim 边界

v3 manifest/protocol/summary/integrity JSON 共核对 8 个 `performance_claimed` / `tir_admitted`
字段，全部为 false。memo、claims map、current status、README、修复记录与 handoff 均只主张 B4-B1
typed reference correctness/gradient parity；未发现 B4-B2/TIR、性能、显存、whole-core/query 或
ASPLOS-ready claim 漂移。

## 5. Findings 与不可现场复核项

- blocker：0
- major：0
- minor：0
- info：0

不可现场复核项仅为当前进程无 CUDA，故没有重新执行 executor 声明的 RTX 4060 CUDA 条件测试；
其收集/skip 边界已如 AC5 所述独立核对。该限制不影响 B4-B1 pure-PyTorch frozen raw replay、F1/F2
直接复核、provenance 或 negative integrity cases。

## 6. 正式处置

- verdict：`approve`
- F1：closed
- F2：closed
- 允许 executor 关闭本 exchange
- 批准后只开放“另行预注册 B4-B2”；B4-B2 实现、CUDA/TIR 与所有性能类 claims 仍须走新的
  预注册、artifact、测试和外审门禁
