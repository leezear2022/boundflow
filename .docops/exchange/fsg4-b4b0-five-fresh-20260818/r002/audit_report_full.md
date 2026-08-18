# FSG4/B4-B0 Five-Fresh Round 2 独立外部审计

- artifact source：`422a3ee96fe86d09bcb0f042b3757447ed94ae6a`
- delivery result：`d5c368c`
- repository HEAD：`e922709`（仅增加 Round 2 exchange 文档）
- verdict：`approve`
- findings：blocker=0，major=0，minor=0，info=0

## F1 关闭结论

Round 1 F1 已关闭。审计方未采信 executor summary，而是从历史 Git blob、冻结 source PT、model、
v2 raw PT/JSON 和现行 verifier 源码独立重算。

- 历史 source `422a3ee` 中的 `FROZEN_SOURCE_IDENTITY` 与 v2 protocol 的完整对象逐字段相等；
  manifest 以 canonical hash
  `05b926ac8fc70f03ce6bd08a34b61ef6bf81cb27e02b019c0cb42c2c590c3e9d` 绑定该对象。
- manifest 与 protocol 的 `source_git_head`、6 个 `code_revision` digest 完全一致；6 个 digest 均由
  `git show 422a3ee:<path>` 的历史 blob 独立重算。
- source capture/model SHA256 分别为
  `f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc` 与
  `791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d`，与本地冻结输入及代码常量一致。
- 从冻结 source capture 与 model 重新初始化 production mapping/plan/schedule，独立衍生的
  source-state、primal-graph、split-state、topology、schedule hash 均与代码常量、protocol 和全部 run
  一致。
- 直接从 source PT 的 raw tensor 重算逐 anchor 的 14 个 source tensor hash；从初始化 mapping 的
  receipt payload 重算 4 个 round-trip receipt hash，全部与代码/protocol及 10 个 capture 的 lineage
  一致。

## AC1 — PASS：provenance 与完整 hash 链

- manifest 精确绑定 14 个非 manifest 文件，逐文件 SHA256 无 mismatch；canonical manifest hash=
  `27391e66acb6fc1146a6fc3f0d726a1b97d24af3df6b24f7294b362e4025be6b`。
- protocol hash=`2514fc21ca34a3647bed5df3352ccebe6dfe07d30147bb9ff2781a003b57ea4b`；
  summary hash=`db7f498c780ef722c182fa1db6d6e1d2baae29b96da73a8d0d644826f8e4413e`。
- 独立重算 10 个 anchor hash、10 个 lineage hash、10 个 capture hash 和 135 个 raw tensor
  dtype/shape/content hash，全部闭合。
- v2 JSON、日志、README 与正式 integrity report 未发现 `/home/` 或 `/tmp/` host-local path。

## AC2 — PASS：five-fresh capture envelope

- 5 个 PT 的 run index 精确为 `0..4`，每 run 两个 capture，顺序均为 S semantic、P performance，
  总计 10 个 capture。
- 5/5 run evaluation/update=`10/9`；10/10 capture evaluation ordinal=`0`、capture count=`1`、
  provider/fallback/eager-backward-fallback=`0/0/0`。
- 10/10 capture 均为 default CUDA stream、priority=`0`、alias pairs 为空；5 个 environment 完全一致，
  device index=`0`、compute capability=`[8,9]`。

## AC3 — PASS：raw 数值、符号与 beta ownership

- 不调用被审计 summary helper，直接比较 run 1..4 与 run 0 的 raw tensor：共 108 tensors、
  664744 elements；最大绝对差=`1.1920928955078125e-07`，全部满足 `atol=rtol=2e-4`，sign exact。
- 每次 S 比较=`14 tensors / 113686 elements`，四次最大差均为 0；每次 P 比较=
  `13 tensors / 52500 elements`，最大差依次为
  `7.450580596923828e-09 / 1.1920928955078125e-07 / 5.960464477539063e-08 /
  5.960464477539063e-08`。
- S：production beta shape=`[6,1]`、numel=`6`，存在 `relu_pre_add_coeff_l` 与 `native_beta`
  gradient，不存在 `incoming_lower_a` gradient。
- P：production beta shape=`[6,0]`、numel=`0`，不存在伪造 pre-add/native-beta gradient，存在
  `incoming_lower_a` gradient。

## AC4 — PASS：integrity negative cases 与 F1 对抗复核

- v2 root replay 通过；正式 probe 由审计方重新执行，11/11 本地 integrity negative cases 全部语义拒绝：
  state、start-node、topology、shape、alpha-index、beta-location、gradient、alias、stream，以及
  coordinated-all-runs topology/lineage。
- 审计方另行构造两种更强的协调一致改写：分别同步改写全部 5 run/10 capture 的 topology，或逐
  anchor lineage source tensor hash；随后重签 lineage/capture、5 个 PT digest、protocol、summary、
  replay stdout、frozen identity hash、manifest file inventory 与 manifest hash。两案 root replay 均
  exit=`1`，在 protocol 对代码冻结身份的绝对校验处以 `FSG4/B4-B0 protocol differs` 拒绝。
- 旧 v1 artifact 以现行 verifier 只读 replay 通过，保持原 summary hash
  `93b62ce30830ffb199f5fc8ddad6db61aff790b4925bccfc18afa238221f399c`，未放宽 v2 身份门禁。

## AC5 — PASS：测试、静态与 DocOps 门禁

- targeted：`24 passed in 2.74s`。
- full：`1376 passed, 3 skipped, 6 warnings in 443.33s`。
- Black：审计范围 7 个 Python 文件 unchanged。
- scoped Mypy：以 `--explicit-package-bases` 检查 capture runtime、worker、artifact runner、integrity
  probe，4 个 source files clean。
- scoped Pylint：上述 4 文件加 artifact test，`10.00/10`。
- `git diff --check a1c6051 d5c368c`、`dol exchange validate` 与 `dol lint --soft` 均通过。

## AC6 — PASS：claim boundary

- manifest、protocol、summary、replay stdout、5 个 PT 与 integrity report 中的
  `performance_claimed`/`tir_admitted` 字段均为 false。
- 计划、claims map、执行备忘和 current status 均明确：Round 2 批准前 B4-B1/B4-B2/TIR/
  performance 关闭；本证据不支持 TIR correctness、速度、显存、系统级或 ASPLOS-ready claim。
- 本轮批准仅关闭 B4-B0 capture correctness/ownership，并只允许下一阶段另行预注册和实施 B4-B1
  typed pure-PyTorch reference。B4-B2、TIR 与所有 performance/memory claim 继续关闭，必须通过后续
  独立门禁。

## Findings

- blocker：无
- major：无
- minor：无
- info：无

## 最终结论

`approve`。AC1–AC6 全部通过，Round 1 F1 的协调一致身份改写缺陷已由 code + protocol + manifest
三层绝对绑定关闭。批准关闭 B4-B0，并仅开放另行预注册的 B4-B1 typed reference；不批准 B4-B2、
TIR、performance、memory 或 ASPLOS-ready claim。
