# FSG4/B4-B0 Five-Fresh Round 1 独立外部审计

- artifact source：`1dbb2de4bc29eb92457e2d24c3e627d638b6607a`
- closure：`a1c6051`
- verdict：`reject`
- findings：blocker=0，major=1，minor=0，info=0

## AC1 — PASS：provenance

- 由历史 Git blob 独立重算 6 个 `code_revision` SHA256，全部与 protocol/manifest 一致。
- manifest 精确绑定 14 个非 manifest 文件，0 digest mismatch；manifest/protocol/summary hash 分别为 `79059be7...01ab`、`d9e8a76e...0da1`、`93b62ce3...99c`。
- source capture SHA256=`f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc`；model SHA256=`791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d`，均与本地冻结文件一致。
- artifact 与 tamper report 的 `/home/`、`/tmp/` 扫描为 0 hit。

## AC2 — PASS：fresh capture 与 ownership envelope

- 5 个 run PT，run index=`0..4`；源码冻结为每个 run 单独启动 subprocess；每 run 精确 2 个 capture，顺序均为 S semantic / P performance，总数 10。
- 5/5 run 均为 evaluation/update=`10/9`；10/10 capture evaluation ordinal=`0`、default stream=`true`、stream priority=`0`、alias pairs为空。
- provider/fallback/eager-backward-fallback 均为 0；环境、source state、schedule identity 在 5 run 间一致。

## AC3 — PASS：raw tensor 与 S/P beta ownership

- 直接加载 raw PT，不调用被审计 summary helper；逐 capture 重算 capture/anchor/lineage hash及每个 tensor 的 dtype/shape/content hash。
- 将 run 1..4 与 run 0 比较：108 tensor、664744 elements；最大绝对差=`1.1920928955078125e-07`，全部满足 `atol=rtol=2e-4` 且 sign exact。
- 每次比较：S=`14 tensors / 113686 elements`，P=`13 tensors / 52500 elements`。
- S：production beta shape=`[6,1]`，存在 `relu_pre_add_coeff_l` 与 `native_beta` gradient，不存在 incoming-lower-A gradient。
- P：production beta shape=`[6,0]` 且 numel=0，不存在伪造 pre-add/native-beta gradient，存在 incoming-lower-A gradient。

## AC4 — FAIL：integrity negative cases

- 原有 9 类本地 integrity negative cases 独立重跑，9/9 均拒绝；它们均修改 run 0 的单个 capture 并重签外层 digest。
- 新增 coordinated topology case：同步将全部 5 个 run、10 个 capture 的 `topology_hash` 改为合法但错误的 `b*64`，重算每个 capture hash、5 个 PT file digest 与 manifest hash。root replay **exit 0，返回 `replay-passed`**。
- 新增 coordinated lineage case：同步改写全部 run 中每个 anchor 的一个 `production_lineage.source_tensor_hashes` 值，重算 lineage hash、capture hash、PT digest 与 manifest hash。root replay同样 **exit 0**。
- 根因：`_discrete_projection` 只比较 fresh runs 相互一致；capture validation 对 topology/source tensor hash 仅检查 SHA256 形状或路径集合，没有把它们绑定到 protocol/source capture 的冻结预期。因此单 run case 被跨-run不一致拒绝，但协调一致的身份伪造可通过。

## AC5 — PASS：validation

- targeted：`20 passed`。
- full：`1372 passed, 3 skipped, 6 warnings`。
- Black：7 个审计范围 Python 文件通过。
- scoped Mypy：B4-B capture runtime + worker/artifact/integrity scripts 共 4 个文件通过。
- scoped Pylint：上述 4 文件加 artifact test，`10.00/10`。
- `git diff --check 1dbb2de a1c6051` 通过。
- `dol exchange validate` 与 `dol lint --soft` 均通过。

## AC6 — PASS：claim 边界；但不得开放 B4-B1

- protocol、manifest、summary、5 个 PT 与 tamper report 共检出 13 个 `performance_claimed`/`tir_admitted` 字段，全部为 false。
- 当前证据只意图支持 B4-B0 capture correctness/ownership；不支持 TIR、region/core/query performance 或 memory claim。
- 因 AC4 major 未关闭，本轮不得批准 B4-B0，也不得据此开放 B4-B1；B4-B2 TIR 与 performance 必须继续关闭。

## Finding

### F1 [major] coordinated source/topology identity 未被语义 replay 绑定

位置：`scripts/run_fsg4_b4b_five_fresh_artifact.py` 的 `_discrete_projection` / `_summary`，以及 `boundflow/runtime/fsg4_b4b_production_region_capture.py` 的 capture/lineage validation。

复现证据：上述 coordinated topology 与 coordinated lineage 两个本地 integrity negative cases 均在逐层重签后被 root replay 接受，exit 0；原 9 类只修改一个 capture，不能覆盖该情况。

建议：在 protocol/manifest 中绑定每个 anchor 的冻结 topology 与 lineage/source tensor identity，replay 应从绑定的 source capture 或不可随 PT 一起重签的 expected projection 验证每个 capture；新增至少 coordinated-all-runs topology 与 lineage-source-hash 两类测试。修复后重新生成 clean-source five-fresh artifact 和 integrity report，再开 Round 2。

## 最终结论

`reject`。数值、符号、S/P ownership、回归和静态检查均通过，但 provenance-sensitive coordinated integrity case 可被接受，违反 AC4。不同意本轮关闭 B4-B0，也不同意开放 B4-B1；B4-B2 TIR/performance 继续关闭。
