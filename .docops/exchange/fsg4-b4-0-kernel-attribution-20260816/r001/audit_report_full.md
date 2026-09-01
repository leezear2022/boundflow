# FSG4/B4-0 Kernel Attribution 外部审计报告(Round 1)

- task: fsg4-b4-0-kernel-attribution-20260816
- auditor: external-model(独立外部审计)
- date: 2026-08-16
- source: `66154e485594e8a84ad1ce04d701d8543c1a7335`
- result commit: `dcd128c`(HEAD `6f07f64` 的祖先)
- artifact: `artifacts/fsg4-b4-kernel-attribution/resnet2b-prop0-v1/`
- 审计原则:不采信 closure/summary 任何数字,全部从 formal raw 用标准库 `gzip/json/hashlib` 独立重算。

## 总体 verdict

**approve-with-findings**(无 blocker、无 major;2 项 minor、2 项 info)。

同意关闭为 `VALIDATED-B4-0-OPPORTUNITY`,并同意批准后只开放 B4-A cumulative candidate;
B4-B 可设计但不得合并执行,B4-C/D 与 B5—B7 继续关闭。

## AC1 — Source/protocol/raw identity:PASS

独立证据:

- `git rev-parse HEAD` = `6f07f6406737610ceacca4918ec091a4b037d785`;
  `git merge-base --is-ancestor` 确认 `dcd128c` 是 HEAD 祖先、`66154e4` 是 `dcd128c` 祖先;
  `git diff --stat dcd128c..6f07f64` 仅 6 个文件、全部为 `.docops/` exchange/ledger 文档改动。
- protocol `source_git_head` 恰为 `66154e485594e8a84ad1ce04d701d8543c1a7335`。
- 10 个 code blob 逐一用 `git show 66154e4:<path> | sha256` 独立重算,全部 OK
  (含 `boundflow/runtime/fsg4_b4_kernel_attribution.py` = `fde094c6…`、runner = `123967fb…`)。
- B3 manifest 绑定:`sha256sum artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1/manifest.json`
  = `d88eeeca…` 与 protocol `b3_formal_manifest_file_sha256` 一致;其 `manifest_hash` =
  `d553a72e…` 与 protocol `b3_formal_manifest_hash` 一致。
- 外部仓库 commit 独立核对:alpha-beta-CROWN HEAD = `e5c7e17bf0488843acb77b7519f59876717a49f4`,
  其 `auto_LiRPA` = `5a098e8f9fb5786a428a024981d833d303921f2d`,vnncomp2021 =
  `90419aadcf06cf543ce5c1706cae1059dc9fa6cf`,三者与 protocol 完全一致。
- 模型/property:`sha256sum resnet_2b.onnx` = `791aa24d…`、`prop_0_eps_0.008.vnnlib` =
  `89edf066…`,与 protocol 一致。
- 多层 hash 链(全部自算,不用被审计方 helper):
  - manifest 内 13 个文件 sha256 全部匹配;
  - `manifest_hash` 重算 = `8720d2f9c8bb2260c1b7a8e9c328762c2a86623b36c5db3ef165825c5891c4b3` ✓;
  - `protocol_hash`、`control/profile worker_hash` 重算均匹配;
  - manifest→protocol、manifest→summary 绑定一致;
  - 解压后 JSONL sha256 = `7f285e27…`、canonical raw hash = `b9a7ff9f…`,均与 worker 绑定一致(见 AC3)。
- 路径泄漏:`grep -rln "/home/\|/tmp/\|file://\|\Users\"` 对 artifact 全部文本(含 gunzip 后
  270,609 行 events)零命中;logs 中路径已被消毒为 `$ABCROWN_ROOT`/`$VNNCOMP_ROOT` 形式。

## AC2 — Fresh semantic pair:PASS

独立证据(从 `workers/control.json` / `workers/profile.json` 重算):

- 两 worker 为独立 fresh 进程(control pid 395354、profile pid 395466),日志时间戳
  01:05:25 → 01:05:51 顺序执行;protocol `worker_sequence` = `["control","profile"]`;
  两者 `configuration=B3`、`mode=control`、`run_id=b4-0-control/b4-0-profile`。
- discrete 字段(batch_size、depths、final_decision、history_count、lower_shape、n_splits、
  n_verified、queue_*、split_depth、status、upper_*)逐项相等,diffs=[];status 均 `verified`。
- lower max abs diff 独立重算 = `4.76837158203125e-07`,≤ protocol 绑定的 B3 冻结
  `atol=rtol=2e-4`(`boundflow/runtime/fsg3_same_solver_timing.py:20-21` 确认
  `FSG3_FLOAT_ATOL=FSG3_FLOAT_RTOL=2.0e-4`);sign exact 独立成立。
- profile/control 扰动披露:core = `410311892/250037760 = 1.6409997114035897x`,
  query = `1775309239/1416088345 = 1.2536712453487497x`,只在 `profile_over_control` 披露。
- `performance_claimed=false` 在 protocol、两个 worker、B3 envelope、summary、manifest、
  tamper report 全部层级成立;control worker `event_count=0`,其 JSONL sha256 恰为空内容
  sha256(`e3b0c44…`)。

## AC3 — 14-call phase closure:PASS

仅用标准库 `gzip/json/hashlib` 独立解析 `events/profile.jsonl.gz`:

- 行数 = **270,609** ✓;解压 sha256 与 canonical raw hash 均与 worker 绑定一致(见 AC1)。
- `event_kind` 分布:cpu_op = 235,201;**cuda_kernel = 35,367**;phase_device_total = 41。
- phase closure:**35,367/35,367 归因,unattributed = 0** ✓。
- attribution method(仅 cuda_kernel):**cpu_parent(correlation)= 33,060,
  temporal_marker = 2,307**,device_marker = 0、unattributed = 0 —— 与 summary
  `phase_closure.attribution_method_counts` 完全一致。
- `event_ordinal` 为 0..270,608 的完整排列,无重复。
- 14 个 CROWN ordinal 独立重建精确成立:`optimizer.crown.00..09`(10)、
  `terminal_export.crown.00`(1)、`kfsb.crown.00..02`(3);4 个 forward ordinal:
  `optimizer.forward.00`、`kfsb.forward.00..02`。distinct phase 共 31 个。
- CUDA user annotation 隔离:41 条 `phase_device_total`(含 `boundflow::b4::*` marker 与
  `Optimizer.step#Adam.step`)独立于 `cuda_kernel` 计数,未被误算为 kernel;
  `test_extract_profiler_events_separates_cuda_annotation_from_kernel` 覆盖该区分。
- 2,307 个 temporal fallback 全部显式标记 `attribution_method="temporal_marker"`,parent 仅为
  `boundflow::b4::worker`(1,263)与 `boundflow::b4::optimizer`(1,044);
  我用 raw 中 marker cpu 行的时间区间独立复核包含关系,**2,307/2,307  containment 成立、
  0 违规**。关键:**全部 14 个 CROWN phase 与 4 个 forward phase 内 0 条 temporal fallback**,
  即 opportunity ledger 完全由 correlation 证据构成。
- 可解析性:0 个 bad stream(kernel stream ∈ {7,13,29})、0 个 null parent、0 个非法
  allocation delta、0 个非法时间区间;input_shapes 在 cpu_op 上 182,813/235,201 带形状
  (materialization 行 31,138 条带形状),CUDA kernel 行 input_shapes 为空(见 minor-1)。

## AC4 — Opportunity 计算:PASS

从 raw 独立聚合(不用 summary helper):

| candidate | calls | kernels | CUDA kernel-sum | materialization ops | device alloc delta |
|---|---:|---:|---:|---:|---:|
| optimizer CROWN | 10 | **6,657** | **24,381,988 ns** | **2,340** | **25,512,960 B** |
| terminal export CROWN | 1 | **578** | **1,117,837 ns** | **252** | **2,851,328 B** |
| KFSB child CROWN | 3 | **1,961** | **7,118,504 ns** | **699** | **28,928,512 B** |
| CROWN14 合计 | 14 | **9,196** | **32,618,329 ns** | **3,291** | **57,292,800 B** |

与 closure 逐项一致;summary 的 `phase_attribution`/`materialization_attribution` 与我的
raw 聚合完全一致。

冻结 wall-share 换算核到 B3 formal raw(`profile_spans.jsonl` + `worker_runs.jsonl`,
6 个 B3 profile worker):

- geomean(core_wall/query_wall)= **0.17735758999613638**(逐位一致);
- geomean((optimizer+backward+kfsb)/query)= **0.12010163988903595**(逐位一致);
- geomean(optimizer/query)= **0.07933101562082898**(逐位一致);
- CROWN14/core = 0.12010163988903595/0.17735758999613638 = **0.6771722591159042** ✓;
- B0 parity target = 1/0.9100012637918488 = 1.0988995727688762x;
- required_r(CROWN14)= **3.989702826086512x** ✓;optimizer-only infinite = 1.086167x 且
  required_r = None(不可达)✓;whole-core infinite = 1.215595x、required_r = 2.030219x ✓。

准入判断:

- **B4-A 成立**:raw 证明 `terminal_export.crown.00` 是完整独立的一次 CROWN call(578 kernel、
  1.118 ms kernel-sum);其 36 个 kernel name 集合与 `optimizer.crown.09` 的 36 个完全相同
  (⊆ 且等势);时序上 Adam 最后一次 update 结束早于 `optimizer.crown.09` 开始(第 10 次
  evaluation 无后续 update),terminal export 在其后执行。即"消除一个完整重复 CROWN call"
  的准入分支有 raw 级结构证据。注意:raw 证明的是结构性重复;lower/lA 数值复用的正确性
  属于 B4-A 自身 five-fresh 门禁,本审计不要求也不声称其已成立。
- **B4-B 成立(仅准入)**:`0.6772 ≥ 0.05` 门槛大幅超过。67.72% 是 span 级 wall share
  (optimizer+backward+kfsb 整段);14 次 CROWN call 的 kernel-sum 占该区 region kernel-sum
  的 68.3%(32.62/47.75 ms),其余为 forward trace build 与 KFSB score/topk 等。即便按更保守
  口径,5% 门槛结论不变。closure 已明确披露换算来源("按冻结 B3 wall-share 换算"),未将
  kernel-sum 升级为 wall speedup、未将 allocation delta 升级为 memory saving(closure §3
  明确否定)。profile wall(扰动 1.64x)未参与任何换算。

## AC5 — Replay / tamper:PASS

- 独立重跑 `python scripts/run_fsg4_b4_kernel_attribution.py replay`:输出
  `{"cuda_kernel_count":35367,"performance_claimed":false,"status":"measured-attribution-only","summary_hash":"987f756db1a257877fbc1581cda85cc00f5d4e7312ab6f3219ad74d58f26bc9e"}`,
  与 artifact `replay_stdout.txt` 逐字节一致;replay 内部逐字段重算 summary(语义对、marker
  计数、raw 重绑定、`derive_b4_attribution`)并与存储 summary 全等比较。
- 独立重跑 `tamper` 子命令:**9/9 rejected**,report_hash =
  `0710e26ceed6d2623bc674978d66b6874a911ef361602bc58a927601b3a7865e`,与 artifact
  `tamper_report.json` 逐位一致(确定性)。代码审读确认每个 probe 先变异、再
  `_resign_outer_artifact` 重算 protocol_hash 与 manifest 全部文件 digest,随后 `_replay`
  仍拒绝——拒绝来自语义重算而非 digest 失配。
- 审计方自建第 10 类变体(不在 executor 9 类之内):把一条 `aten::empty` 的
  `device_memory_delta_bytes` +4096,然后正确重签 worker_hash、event_count、raw hashes 与
  manifest 全部内外层 digest,replay 仍拒绝:`ValueError: FSG4/B4 summary replay differs`。
  证明拥有完整重签能力的攻击者也无法通过。

## AC6 — Regression/static/DocOps:PASS

- B4 targeted:`15 passed`(test_fsg4_b4_kernel_attribution.py 8 +
  test_fsg4_b4_kernel_attribution_runner.py 7)。
- B3/B4 相关:`54 passed`(same_solver 三件套 25 + explicit_counters 14 + B4 15;
  该 54 的文件构成在 request/closure 中未固定,审计按最合理构成复现;全部 16 个
  fsg4_b3/fsg4_b4 文件并集为 `96 passed`,亦全绿)。
- 全量:`1329 passed, 3 skipped`(477s 完整重跑)。`-rs` 核对 skip 理由:
  `test_artifact_phase5d_smoke.py`(TVM 可用时主动跳过免重复编译)、
  `test_cross_axis_verification_batch_artifacts.py` 与 `test_root_projection_floor_artifacts.py`
  (frozen VNN-COMP checkout 路径不可用)——均为环境性/成本性 skip,与 B4 无关。
- `black --check`(4 个改动文件)、`mypy`(2 模块,无 issue)、`pylint` **10.00/10**、
  `git diff --check` 全部通过。
- `dol exchange validate fsg4-b4-0-kernel-attribution-20260816` 与
  `dol exchange validate fsg4-b3-formal-timing-20260814` 均 `{"ok":true}`;
  `dol lint --soft` = ok。
- 工作树仅 `M .docops/ev.jsonl`、`M .docops/exchange/.../state.json`(流程自动改动)。

## AC7 — Claim 边界:PASS

- closure/prereg/claims map/备忘录/handoff 五处一致:只有 attribution/opportunity claim,
  `performance_claimed=false` 贯穿所有层级;无 B4 speedup、无 B0 parity 达成、无 memory
  saving、无 ASPLOS-ready("memory saving"/"speedup" 仅以否定或 required-r 门槛形式出现)。
- 路由:approve 后只开放 B4-A(terminal lower/lA handoff,先过 5 fresh correctness 与
  B3/B4-A core≥1.03x/query worst≥0.98x 门禁);B4-B 可设计但不得与 B4-A 混跑;B4-C/D、
  B5—B7 继续关闭。`.docops/s.md` blocker=`awaiting-external-audit`、exchange state=`auditing`,
  与本审计流程一致。

## Findings

- **minor-1**:CUDA kernel 行 `input_shapes` 全部为空(0/35,367),形状只在 cpu_op 行
  (182,813/235,201)。prereg §6 要求输出包含 "input shape";operator/materialization 层
  已满足,kernel 层形状缺失是 torch profiler 的固有行为,不影响归因与 opportunity 结论,
  但 B4-B 冻结 production shape 时需从 operator 行取形状。建议:在 B4-B 开工文档中显式
  记录"kernel 形状从 correlation parent operator 行恢复"。
- **minor-2**:`54 passed` 的 "B3/B4 相关" 文件集合未在 request 或 closure 中固定。
  审计以 `same_solver×3 + explicit_counters + b4×2` 复现恰为 54;若定义不同(如全并集 96),
  数字不同但同样全绿。建议:后续 exchange 在 request 中固定 related 集合的文件清单。
- **info-1**:closure "CROWN14 覆盖约 67.72% B3 core" 的 67.72% 是 span 级 wall share;
  14 次 CROWN call 的 kernel-sum 占该 region kernel-sum 的 68.3%,region 内还含 forward
  trace build/KFSB score 等非 CROWN 工作。文档已披露换算来源,且 5% 门槛在保守口径下
  结论不变,不构成 claim 漂移。
- **info-2**:B3 span 时序显示 `kfsb.crown.01/02` 的 CUDA 时间戳早于 `optimizer.crown.00`,
  系 CPU/CUDA 时钟域混合所致;ordinal 归属全部基于显式 marker,不受影响。

## 不可现场复核项

- 两次 fresh worker 的原始物理执行过程(温度、独占性)只能依据 artifact 内 preflight/
  environment 记录与日志,审计无法重放当时机器状态;记录内部自洽(交流供电、温度 50—54°C、
  仅 kwin_wayland 与本 worker 进程)。
- B4-A 的 lower/lA 数值复用正确性不在 B4-0 范围,留待 B4-A five-fresh 门禁。

## 关键命令与输出摘录

```text
# AC1 hash 链
file digests all match: True count: 13
manifest_hash recompute match: True 8720d2f9…c4b3
all code_revision match: True(git show 66154e4:<path> 逐一 sha256)
b3 file sha match: True / b3 manifest_hash match: True
grep 本机路径(含解压 events):0 命中

# AC2
discrete diffs: [] ; lower max abs diff: 4.76837158203125e-07 <= 2e-4: True ; sign exact: True
core ratio: 1.6409997114035897 ; query ratio: 1.2536712453487497

# AC3(标准库独立解析)
line count: 270609 ; cuda_kernel: 35367 ; unattributed: 0
attribution methods: {'cpu_parent': 33060, 'temporal_marker': 2307}
crown ordinal set exact: True ; forward ordinal set exact: True
temporal containment violations: 0 of 2307 ; CROWN/forward phases 内 temporal fallback = 0

# AC4
CROWN14 total: kernels 9196, ns 32618329, mat_ops 3291, dev_alloc 57292800
geomean core/query=0.17735758999613638, crown14/query=0.12010163988903595 → 0.6771722591159042
required_r(crown14)=3.989702826086512 ; optimizer-only required=None
terminal_export.crown.00 kernel names ⊆ optimizer.crown.09(36=36);adam 最后 update < crown.09 开始

# AC5
replay 输出与 replay_stdout.txt 逐字节一致,summary_hash=987f756d…
tamper 独立重跑 9/9 rejected,report_hash=0710e26c… 与 artifact 一致
审计方第 10 类变体(alloc delta +4096,全量重签)→ ValueError: FSG4/B4 summary replay differs

# AC6
15 passed(targeted) / 54 passed(related) / 96 passed(fsg4_b3+b4 并集)
full: 1329 passed, 3 skipped
black ✓ / mypy ✓ / pylint 10.00/10 / git diff --check ✓
dol exchange validate ×2 → {"ok":true} ; dol lint --soft → ok
HEAD == origin(6f07f64);dcd128c..6f07f64 仅 .docops 文档
```
