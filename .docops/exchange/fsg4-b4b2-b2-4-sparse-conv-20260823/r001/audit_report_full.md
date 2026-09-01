# B4-B2 B2-4 Sparse-source Conv P0 + 12 候选 Pre-timing Ledger 外部审计报告(Round 1)

- task: `fsg4-b4b2-b2-4-sparse-conv-20260823`
- auditor: external-model(独立外审,Round 1)
- branch: `feat/rvir-v4-production-state-ownership-v1` @ `5584999`(HEAD==origin,已核实)
- base: `b18fad483fcfa9bbef61337628f368a7ca2fd7c2`(B2-3 外审正式关闭)
- source/result: `1f8d47a8acd55f9b315e207a549f515e29a6f35e`
- 审计时间: 2026-08-23(UTC)
- 环境: conda `boundflow`,RTX 4060 Laptop GPU(sm_89),全部 runner/oracle/pytest 现场重跑

## 总体 Verdict

**APPROVE**(0 blocker / 0 major / 2 minor / 2 info)。

B2-4 的 P-anchor sparse-source Conv P0 correctness 与 12 项预注册 bounded schedule
candidate ledger 经独立复核全部成立;claim 边界无漂移;同意关闭 B2-4 并仅开放 B2-5
(formal independent-process artifact/replay/AB-BA timing,必须复用本次冻结的 12 项
ledger,不得追加第 13 候选)。B4-B3、任何 speedup/winner/memory/ASPLOS-ready 表述仍关闭。

## 审计方法声明

未采信交接方任何汇总数字。所有关键事实均由本人现场独立复核:

1. `git show`/`git diff` 核对 commit 范围与预注册门禁条文;
2. 直接解析 5 份 `run_XX.pt` production raw capture(不经过 repo runtime);
3. 以 numpy float64、无 autograd 的闭合公式独立实现 sparse Conv forward/backward
   oracle(从 TIR 源码独立推导语义,不复用 repo 的 B4-B1 reference),与 GPU TIR 输出
   逐元素比对;
4. 现场重跑 `scripts/run_fsg4_b4b2_sparse_conv_tir_correctness.py`(P0 five-raw +
   12 候选 confirmation);
5. 独立重算 template/schedule/ledger hash,亲手做 12 项篡改 fail-closed 探针;
6. 直接遍历 12 个 scheduled TIR script 的 `alloc_buffer`/thread_binding/unroll 结构;
7. 现场跑 targeted/related/full pytest 与 black/mypy/pylint/TVM rebuild/DocOps。

## AC1 — Git 顺序、范围与预注册:PASS

- `1f8d47a` 直接后继于已 approved/closed 的 B2-3 commit `b18fad4`;`5584999` 为 handoff
  文档提交,HEAD==origin。
- `git show --stat 1f8d47a`:17 个文件、+2920/-8;代码改动恰为交接声明的 6 个文件
  (3 source + runner + 2 test),其余为文档/DocOps;无 production、optimizer、vendored
  TVM/TVM-FFI/auto_LiRPA 改动;无任何 timing/B2-5/B4-B3 实现代码。
- 预注册门禁条文无事后改动:`git diff b18fad4 1f8d47a --
  gemini_doc/BOUNDFLOW_FSG4_B4B2_TYPED_CUDA_TIR_PREREGISTRATION_PLAN_2026_08_23.md`
  的全部删除行仅 frontmatter `status`/`updated` 两行;门禁段落(knob 集合、≤12 冻结、
  不得追加候选)逐字未动,新增内容均为追加的结果小节。
- `performance_admitted=false`、`timing_raw_present=false`、`winner_selected=false`、
  `performance_claimed=false` 在 receipt `validate` 中以 `is not False`/`is not True`
  硬校验(fail closed),非仅文档声明。

## AC2 — Production compressed-α 映射与 empty-β absence:PASS

直接解析 5 份 raw capture(`payload["captures"][1]`,`operator_kind=conv2d`):

- `production_alpha=[2,1,6,86]`,合同选择 `[0,0]` 得 compressed α `[6,86]`;
  5 个 run 的 α 值域均在 `[0,1]`(实测 min=0.0/max=1.0);
- 三个 `feature_index/{0,1,2}` 各 `[86]`(int64),组成 86 个 (c,h,w) 坐标,5 个 run
  均满足:86 个坐标全唯一、全部落在 `[16,8,8]` 范围内,`feature_shape=[16,8,8]`;
- `production_beta=[6,0]`、`beta location=[6,0]`、`beta sign=[6,0]`;raw capture 的
  gradients 只有 `incoming_lower_a` 与 `native_alpha` 两项,**无** `native_beta` 梯度
  (β gradient 真 absent,非零 tensor);
- ABI 结构证据:`SPARSE_CONV_INPUT_NAMES` 恰为 9 个输入,不含任何 β/native-α 项;
  forward primfunc 签名 9 输入+2 输出,backward 8 输入+2 输出,签名中无 β buffer;
  坐标经 `_compressed_alpha_value` 以常量 if-then-else 链 inline 进 TIR,并进入
  `alpha_coordinate_hash`(template stable hash 的组成部分);
- 无 dense α workspace 的结构性证据见 AC6(alloc_buffer 仅两项,script 全文无
  `native_alpha`/`native_beta`/`compressed_beta`/`scaled`/`scatter` 字样)。

## AC3 — 独立数学 oracle:PASS(重点项)

本人从 TIR 源码独立推导语义并以 numpy float64 闭合式实现(脚本
`/tmp/b2_4_independent_oracle.py`,不使用 repo reference、不使用 autograd):

- forward:compressed α 按 86 个常量坐标 scatter 重建 dense α;`slope_up =
  l>=0 ? 1 : (u<=0 ? 0 : u/max(u-l, eps))`(eps=2^-23);`slope_low = amb ?
  clamp(α,0,1) : (l>=0 ? 1 : 0)`;`sel_slope = incoming>=0 ? slope_low : slope_up`;
  ConvTranspose 形式 `out[d,co,oh,ow] = Σ_{ci,kh,kw} relu_a[d,ci,oh+1-kh,ow+1-kw]·
  w[ci,co,kh,kw]`(界内);bias 通道含 `incoming·sel_int + relu_a·op_bias` 双项;
- backward:adjoint 索引 `adj[d,ci,ih,iw] = Σ_{co,kh,kw} g[d,co,ih-1+kh,iw-1+kw]·
  w[ci,co,kh,kw]`(与 forward 线性映射严格转置);`adj_relu = adj + g_bias·op_bias`;
  compressed α grad = `Σ_s [incoming>=0 ∧ amb ∧ 0<=α<=1] adj_relu·incoming`(gather 到
  86 坐标);incoming-A grad = `adj_relu·sel_slope + g_bias·sel_int`(`[6,1,16,8,8]` 全覆盖,
  无坐标限制);
- 结果:5 个 raw × 4 路输出,oracle vs GPU TIR 最大绝对差 **1.830034e-06**
  (float32 归约舍入量级,远低于 atol/rtol 2e-4),全部 finite、sign exact;
- gradient projection 独立复核:raw capture 的 native α gradient 在 516(=6×86)个
  owned 坐标之外**严格为零**(unowned absmax = 0.000e+00);candidate compressed α
  grad 与 raw native grad 按坐标 gather 的差为 2.049e-08;owned count=516 精确。

## AC4 — 现场 P0 five-raw GPU gate:PASS

现场执行 `python scripts/run_fsg4_b4b2_sparse_conv_tir_correctness.py`(RTX 4060,
独立进程),输出 JSON 逐项核对:

- run/metrics/elements = 5/20/64,050 ✓;
- max diff = `2.384185791015625e-06`(5 个 run 各自相同),allclose/sign exact 全 true ✓;
- cache 序列 = `miss,hit,hit,hit,hit` ✓;5 个 run 的 module receipt hash 完全相同
  (`44800f32e23693881cc7515cc8a4048eb005e1893e9a1d97fa4a7cc30851fce4`),即 hit 真实
  复用同一 module ✓;
- template hash = `c51b77cbdf28551cb8b97252d82a5abdda76851c5fb49e0a54547ac898f14075` ✓
  (本人另以独立脚本从 raw capture 重建 template 重算,逐位一致);
- P0 schedule hash = `a4937031…55f1` = ledger ordinal 0 ✓;
- 每 run fwd/bwd launch = 1/1、fallback/eager = 0/0、DLPack = 19/19、β gradient
  absent(launch receipt `beta_gradient_present=false` 硬校验)。

## AC5 — 12 候选 pre-timing ledger:PASS(重点项)

- 候选集合恰为预注册允许集合的 balanced subset:12 个 knob 四轴取值
  (thread{128,256} × octile{4,8,16} × spatial{1,2} × unroll{1,3})逐轴全覆盖,
  无任何越界组合(独立集合运算验证 `knobs ⊆ allowed` 且每轴 cover 完整);
- 12 个 schedule hash 与 12 个 module receipt hash 全部唯一(现场 runner 输出集合
  基数各 12);12 个 scheduled TIR script 两两不同(AC6);
- ledger hash 独立重算 = `1660edca9f23201b14edfe8ce06947ec16f52b5b311ddb47174ea1955e8d07c6`,
  与交接逐位一致;schedule hash 由 `(template, ordinal, knobs, workspace, flags)` 计算,
  ledger hash 由 12 个 schedule hash 计算,链完整;
- 12 候选各在 fresh module cache 下 cache event 全为 `miss`(cache key 绑定
  template+schedule hash,不存在一个 module 冒充 12 个 schedule 的可能);
- 12 候选 × 4 metrics × capture0 = 48 metrics / 153,720 元素,全部 allclose/sign exact,
  全候选 max diff = `2.384185791015625e-06` ✓;
- 五个冻结字段由代码强制而非文档声明:`DifferentiableLowerSparseConvCandidateLedgerV1.
  validate_against` 硬校验 `len(schedules)==12`、ordinal 恰为 0..11、hash 唯一且与
  schedules 重算一致、`generated_before_timing is True`、`timing_raw_present/
  winner_selected/performance_claimed is False`;
- 本人亲手篡改探针 12 项全部 fail-closed:越界 knob(512 线程)、ordinal/knob 错位、
  第 13 候选、ordinal 乱序、四个冻结字段逐一翻转、schedule_hashes 篡改、重复 hash、
  `from_dict` 重签后篡改、ledger 外 knob 组合(128,4,2,1)——均抛 `ValueError`;
- 无暗中排序/winner:runtime 与 runner 全文 grep 无任何 timing/perf_counter/elapsed/
  winner/sort-by-metric 代码,候选按 ordinal 0..11 固定顺序生成,ledger 在 compile
  之前由 knobs 确定性构造。

## AC6 — 结构 workspace 与物理 schedule:PASS

- 直接遍历 12 个 scheduled TIR 的 `Block.alloc_buffers`(并经 script 文本复核):
  每个候选恰为 `adjoint_conv[6,1,16,8,8]` + `output_bias_delta[6,1]`;forward 的
  `output_bias_delta`、backward 的 `adjoint_conv` 之外无任何 alloc;
- `relu_lower_a`/`adjoint_relu` 已 compute_inline(无任何对应 alloc_buffer);
- script 全文无 `native_alpha`/`native_beta`/`compressed_beta`/`scaled`/`scatter`
  buffer 或符号;
- schedule 变换真实反映在 loop 结构:thread_binding extent 恰为 128/256 两组;
  `T.unroll(3)` 仅出现在 ordinal 10/11;blockIdx 维度随 thread extent 变化
  (128→48 块,256→24 块);12 个 script 两两不同且差异位于 loop 结构而非命名。

## AC7 — Receipt 链、负路径与 B2-3 遗留处置:PASS

- Template/Instance/Schedule/Module/Projection/Launch/Ledger 七类 receipt 的
  round-trip 与篡改拒绝在 `tests/test_fsg4_b4b2_sparse_conv_tir.py` 中具体断言
  (duplicate/out-of-range 坐标、shape/dtype/device/nonfinite/α range、fallback/
   eager、higher-order、stream、pointer、launch、projection 516/mapping/unowned-zero/
  β-absent、module/launch 的 performance_claimed 篡改);本人另补 12 项亲手篡改探针
  全部 fail-closed(AC5);
- S-anchor/active-β/scope broadening 的拒绝由两层硬编码保证:direction/spec 选择固定
  `production_alpha[0,0]`,template 常量 `anchor_id="performance-conv-8-candidate"` 在
  `validate` 中强校验,`_mapping_coordinates` 拒绝任何非 `[6,0]` 的 β location/sign;
  S-anchor capture 会因 anchor/shape/mapping 不一致在构造期抛错;
- B2-3 遗留 info #2(dense Conv shape-mismatch 专项用例)确已补上:
  `tests/test_fsg4_b4b2_dense_conv_tir.py` 在 1f8d47a 中新增
  `native_alpha[:, :, :, :-1]` shape-mismatch 拒绝断言,随 targeted 51 通过真实运行;
- B2-3 遗留 info #1(module TIR/device-source hash 重编译比对)**未虚假关闭**:
  changelog 明确写"延期至 B2-5:formal replay 必须独立重编译并比对 TIR/device-source
  hash",delivery risks 同样声明。保持开放,转入 B2-5 验收项。

## AC8 — 测试与静态验证链:PASS

- targeted `pytest -q tests/test_fsg4_b4b2*.py`:**51 passed**(现场,181s)✓;
- related `pytest -q tests/test_fsg4_b4b*.py`:**105 passed**(现场)✓;
- full `pytest -q`:**1465 passed, 3 skipped**(现场,-rs 核对:3 个 skip 均为既有
  环境边界——1×allow-no-TVM 重复编译去重 + 2×frozen VNN-COMP checkout 缺失,与
  B2-4 无关)✓;
- `pytest -q tests/test_env.py`:3 passed ✓;
- black --check 6 文件 clean;mypy 4 source files clean;pylint 三个新 source
  10.00/10;`bash scripts/rebuild_tvm.sh` → `ninja: no work to do`;
- DocOps:`dol lint --soft` 通过;`dol validate` 报 3 个 duplicate event id,经核对
  在 base commit `b18fad4` 即已存在(2026-08-14 与 2026-08-23 凌晨的历史事件),
  非本轮引入(见 info finding F2)。

## Findings

- F1 | minor | `gemini_doc/BOUNDFLOW_FSG4_B4B2_B2_4_SPARSE_CONV_TIR_CHANGELOG_2026_08_23.md:69`
  | 候选 5 的 schedule hash 后缀缩写写为 `6ab7c314…1646`,真实 64 位 hash 末 4 位为
  `646d`(完整值 `6ab7c314f51c7f4e193e777568e2730a51d3aa501e5aeb2d161d543a05a1646d`,
  作者截取了末 6 位 `a1646d` 的中间 4 位)。ledger 完整 hash 与本人独立重算逐位一致,
  说明真实 schedule hash 集合无误,仅文档表格缩写笔误。
  | 建议:修正该表格后缀为 `…646d`;不影响任何门禁与 hash 链。
- F2 | info | `.docops/ev.jsonl` | `dol validate` 报 3 个 duplicate event id
  (ev009180/ev009388/ev010862),在 base `b18fad4` 已存在,系历史遗留非本轮回归;
  `dol lint --soft` 通过。 | 建议:后续窗口期清理历史重复事件 id,本轮不阻塞。
- F3 | info | `boundflow/runtime/fsg4_b4b2_sparse_conv_tir.py` | module receipt 的
  TIR/device-source hash 在 validate 中仅做格式校验,独立重编译比对依计划留待 B2-5
  replay(B2-3 遗留 info #1 的延续,交接已如实声明,未虚假关闭)。
  | 建议:B2-5 必须将该项列为硬性验收。

## 不可现场复核项

- module TIR/device-source hash 的独立重编译比对:本轮按预注册明确留给 B2-5 replay,
  本轮范围内不可复核也不应复核(不构成 finding 之外的扣分)。
- 长期数值稳定性(超出 5 份冻结 raw 的分布外行为):本轮范围仅覆盖冻结 P0 raw。

## Claim 边界核对

五处权威文档(changelog、prereg plan、claims map、execution memo、master plan +
current_status)新增段落措辞一致:仅声明 P0 correctness 与 12 项 bounded ledger
compile/correct,全部明确"无 timing/winner/performance claim、B2-5/B4-B3 关闭";
无 speedup、memory、whole-core/query、B0 parity、ASPLOS-ready 表述。**无 claim 漂移**。

## B2-5 开放意见

同意开放 B2-5,且仅 B2-5:formal independent-process artifact + replay + 预注册 AB/BA
timing;必须复用本次冻结的 12 项 ledger(hash
`1660edca9f23201b14edfe8ce06947ec16f52b5b311ddb47174ea1955e8d07c6`),不得追加第 13
候选;B2-5 验收必须包含 module TIR/device-source hash 独立重编译比对(B2-3 info #1)。
B4-B3 继续关闭。
