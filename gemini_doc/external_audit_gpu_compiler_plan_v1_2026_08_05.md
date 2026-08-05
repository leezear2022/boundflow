---
status: final
updated: 2026-08-05T15:40:00Z
type: external-audit
topic: boundflow
slug: gpu-compiler-acceleration-research-v1
audited_branch: feat/top2-production-execution-cost-attribution-v1
audited_head: 849912d
---

# 外部审计报告:GPU 编译器加速研究计划 v1(2026-08-05)

审计对象(均为未提交工作树交付物):

- `gemini_doc/BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md`(955 行)
- `gemini_doc/BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_CHANGELOG_2026_08_05.md`(80 行)

审计方式:不信任文档自述,全部事实性声明对照当前源码、历史关闭文档、artifact 原始数据与 git 状态独立复核。

## 总体 Verdict

**approve-with-minor**(0 blocker / 0 major / 1 minor / 2 informational)。

文档的核心立场(收敛主线为 verification-aware selected-CROWN GPU compilation、NRIR48 归因数字、
BoundConv 40x 降级为 USER-REPORTED、Schedule/stream/event/arena 未物理闭环、auto_LiRPA 强 baseline、
下一步只做 G0/G1、performance_claimed=false 与 IR-5 NO-GO 不漂移)经逐项独立复核**全部成立**。
唯一 minor 是 §6 历史表中 PR-12J 一行的 compile 数字归属偏差(见 Findings)。

## 逐项结论

### 1. 数字一致性:PASS

对照 `gemini_doc/change_2026-08-05_nrir48_execution_cost_attribution.md` 与
`artifacts/top2-production-execution-cost-attribution/vnncomp21-resnet2b-property0-three-repeat-cpu-phase0-v1/formal.json`:

| 计划文档声明 | 独立复核值 | 结果 |
|---|---|---|
| child execute 占 queue 32.20%/31.16% | NRIR48 关闭文档 32.1966%/31.1640% | PASS |
| selected-CROWN 每 queue 2.663/2.694 s | formal.json median_ns 2663321156/2694435710 | PASS |
| 占 child execute 71.77%/72.73% | formal.json median_share_of_parent 0.717725/0.727291 | PASS |
| 占 whole trace 约 17.1% | (2.663321+2.694436)/31.319772 = 0.17106 | PASS |
| whole trace median 约 31.320 s | NRIR45 三次 trace 31.262521/31.319772/31.470078,median=31.319772 | PASS |
| 40x region → whole-trace 上限约 1.20x | 1/(0.829+0.171/40) = 1.19996 | PASS |
| child execute 合计约 24.0%,无限加速上限约 1.32x | (3.816002+3.704755)/31.319772 = 0.24013;1/0.76 = 1.3158 | PASS |

墙钟 36.4–36.6 s 出自 NRIR45 文档(36.396631/36.513683/36.611709),被审计划未引用,无冲突。
formal.json `performance_claimed: false`、`dominant_child_subcategory: selected_crown_ns` 与文档一致。

### 2. 代码事实:PASS(逐行号复核)

- `boundflow/runtime/native_intermediate_refinement.py:923` `_run_selected_crown`:按 ReLU 分组、
  按 chunk 切块(`policy.backward_chunk_size`,各处实例化默认 32)、每 chunk `torch.zeros` 物化 dense
  one-hot(L945-950)、构造 indices tensor(L953)、逐 ReLU 逐 chunk 调
  `run_crown_ibp_mlp_from_forward_trace`(L955)、**未传** fused_crown_executor/steps/context、
  末尾 `torch.cat`(L967-968)、`_intersect_selected` 与 intermediate bounds 相交 —— 全部属实。
- `boundflow/runtime/crown_ibp.py:2136` `run_crown_ibp_mlp_from_forward_trace` 签名确有
  `fused_crown_executor/fused_crown_steps/fused_crown_context` 参数 —— "被调用函数已支持这些参数"属实。
- `boundflow/runtime/crown_ibp.py:1563-1570` `_execute_fused_relu_affine_step` 对
  non-plain/requires_grad/alpha/beta/split_state 直接 `return None`(fail-closed);selected-CROWN
  调用点不传 `relu_alpha`(默认 None)—— G2 的 legality 判断属实。
- `boundflow/ir/schedule.py:1240` reference lowering `LaunchAction(..., stream_id="sync")` —— 精确命中。
- `boundflow/runtime/schedule_ir_executor.py:172` `execute_schedule_reference`:RecordEvent/WaitEvent
  仅 `append_event` 记 trace(L299-313),全文无 `torch.cuda.Stream/Event` 引用 —— "只记 trace"属实。
- `boundflow/runtime/storage_plan_runtime.py:206` `StoragePlanRuntime`:last-use 释放 Python env 引用 +
  logical live-byte accounting,无物理 arena/offset view —— 属实。
- `boundflow/runtime/fused_crown.py:279` `_plain_static_cuda_fp32`:要求 plain/no-grad/no-alpha/
  no-beta/no-split + CUDA FP32 contiguous —— "capability 只接受 plain static CUDA FP32"属实。
- `boundflow/backends/tvm/fused_crown_conv2d.py:28/52`:signature dataclass,
  `schedule_id="output_gather_128t_v1"` —— "固定 128 threads"属实。
- `boundflow/backends/tvm/fused_crown_cache.py:44` 起为 validated cache(signature payload + SHA256)—— 属实。
- `boundflow/runtime/query_batcher.py:164` `DynamicBatchManager`;`estimate_request_bytes` 文档字符串
  明确 "without claiming allocator peak" —— "memory estimate 只是 payload bytes"属实。
- vendored auto_LiRPA `operators/convolution.py:70` `BoundConv.bound_backward`:OneHotC 走
  `onehotc_to_dense`(L81-83),Tensor path 用 `F.conv_transpose2d`(L95-98),另有 `Patches` 路径
  (L116)—— §8.1 表述属实;README L26-27 确有 "memory efficient GPU implementation of backward
  (CROWN) bounds for convolutional layers"。
- vendored auto_LiRPA `setup.py` L43-44:`torch>=2.0.0,<2.9.0` —— 与当前 PyTorch 2.12.1 冲突的
  声明属实。

环境声明(§3.1/3.2)现场复核:Python 3.12.12、PyTorch 2.12.1+cu132、
`torch.cuda.is_available()==False`/device count 0、TVM 0.23.dev0 且 `enabled("cuda")==True`、
`nvidia-smi` 无法连接 driver、HEAD=849912d、origin/main=c0ccfb5(HEAD 是其祖先)、三个 submodule
SHA —— 全部一致。

### 3. Claim registry:PASS

12 个 claim ID(F-ENV-01/02、F-IR-01、F-EXEC-01、F-CPU-01、U-40X-01、H-GPU-01、H-FUSE-01、
H-MEM-01、H-SCHED-01、H-JIT-01、C-END-01)逐一可定位;为新增研究级 ID,与
`asplos_claims_map.md` 现有 C1/C2/C3 体系无冲突、无重复定义。状态标注保守:无任何 INFERRED/USER-REPORTED
被标为 MEASURED;无新增隐性性能 claim。BoundConv 40x 全文 17 处出现,逐一检查后全部处于
USER-REPORTED/待复现/条件假设语境(§2.1、§5.2 Amdahl 假设句、§7、§10 G0、§14),无漂移。
G8 中 "C-END-01/C2 升级须 held-out p90 Oracle regret ≤1.20x" 与 IR-5 冻结门槛一致。

### 4. 链接与结构:PASS

- 本地链接:plan 12 个 + changelog 2 个 = 14 个,逐一 `os.path.exists` 验证全部存在
  (§17 的 12 个证据入口 + changelog Links 2 个)。
- G0—G8 九个阶段 DAG 与 kill gate 自洽:G1 `PROPOSED-GATE ≥20%` 未过 → "重新选择 GPU winner /
  关闭本路线"(DAG 与 §14 kill decisions 表第二行一致);G2 qualification NO-GO → 保留 reference
  fallback 进 G3;G6 两级 control(packed vs 最强 single-stream、multi-stream vs 已 qualified packed)
  防止弱 baseline 重复计算 speedup;G7 JIT/Graph 准入依赖 G5 arena 与 measured reuse;G8 依赖全部前序。
  未发现循环依赖或门禁悬空。
- benchmark 矩阵(§11)与 artifact/replay 合同(§12):字段(environment_id、digest、反平衡、
  raw/normalized/summary、tamper 重算)与仓库现有 artifact 惯例方向一致;注意合同文件清单
  (manifest/environment/queries/results_raw/normalized/summary/failure_rows)是**前向预注册**的新
  schema,当前 NRIR48 artifact 实际结构为 formal.json+manifest.json+shards/+logs/,文件名不同——
  属 G0 待实现项而非失实(见 informational finding I-1)。

### 5. 历史 NO-GO 引用:PASS(1 处归属 minor)

| 引用 | 原始关闭文档复核 | 结果 |
|---|---|---|
| PR-12I:8.644/1.736、1.768/1.386、7.009/7.234、geomean 0.546x | change_2026-07-14_pr12i_fair_baselines.md L57-62 | PASS |
| PR-12J:break-even 4668/1062/4450、Q≤1024 FAIL | change_2026-07-14_pr12j_compile_amortization.md L71/87 | PASS |
| PR-12J:"compile 约 0.29–1.48 s" | 实际出自 change_2026-07-13_pr12ef_runtime_pareto_heldout.md L57(PR-12EF 的 compile overhead);PR-12J 自身 compile phase 为 323.67/480.00/1299.12 ms | **minor M-1** |
| PR-12K:launch 最大降 1.96%、3 regress/1 improve/2 neutral | change_2026-07-14_pr12k_cupti_profile.md L57-65 | PASS |
| PR-12L:E_STOP_OPTIMIZING_TIR | change_2026-07-14_pr12l_stop_tir_optimization.md L8 | PASS |
| PR-13D:96.52x / 1.024x / hard 0.980x | pr13_closure_audit L47-49、change_2026-07-14_pr13d_fixed_e2e_gpu.md L37/45 | PASS |
| IR-5:Global p90 regret 1.26160x > 1.20x,VALIDATED-NO-GO | change_2026-07-28_ir5h_residual_final_v3_nogo.md L58/61/110 | PASS |
| NRIR43:launch 31→16、queue 慢约 4%–5% | nrir43 nogo L14-15(ratio 1.051134/1.044573) | PASS |
| NRIR46:static shareable median 1.071 s < 1.5 s | nrir46 nogo L16/28-29(1.071197) | PASS |
| NRIR47:queue 慢约 1%–2% | nrir47 nogo L33-35(10.099396→10.212559、10.056289→10.250753) | PASS |
| "历史 fused coverage 仍是 0/394" | asplos_claims_map.md L22/432/440 | PASS |

### 6. DocOps/索引:PASS

- `dol lint --soft` 现场重跑:`{"ok":true,"miss":[],"soft":true}` —— 与 changelog 自述一致。
- changelog 自述的 DocOps 事件 ev005726/ev005727(scaffold)、ev005813(change,ty=ch 且
  slug=gpu-compiler-acceleration-research-v1)、ev005819(validation,ty=va)在 `.docops/ev.jsonl`
  中均存在。
- `.docops/s.md` diff:blk=gpu-session-unavailable、next=restore-gpu-and-freeze-nrir49-g0-g1-baselines、
  last_ch/last_va 指向最新事件 —— 与新文档主题一致。
- `docs/change_log.md` 与 `gemini_doc/README.md` 均已新增本计划入口条目,内容(40x 未复现、
  research-only、只做 G0)与计划正文一致。
- `git diff --check`(tracked)PASS;两个 untracked 文档以
  `git diff --no-index --check /dev/null <file>` 复核,无 whitespace error 输出(exit=1 仅因
  /dev/null 与文件存在 diff,属预期)。
- `git status`:M .docops/ev.jsonl、M .docops/s.md、M docs/change_log.md、M gemini_doc/README.md +
  两个 untracked 新文档,无其他改动 —— changelog "docs-only" 自述属实。

### 7. Claim boundary:PASS,无漂移

- 全文 grep 越界表述(已验证/快于/outperform/ASPLOS-ready YES/GPU 结果已 等)仅命中 §3.3
  "不能声称"否定句;§3.3 负面清单与 asplos_claims_map 的 `performance_claimed=false`、
  "ASPLOS-ready No-Go 不变"(L16)完全一致。
- IR-5 仍 VALIDATED-NO-GO(§3.3、§6、G8 gate);BoundConv 40x 始终 USER-REPORTED;
  logical storage 1,860,912→442,656 B 明确标注为逻辑 liveness 而非物理 CUDA peak(§3.3)。
- 测试数字:被审两份文档均不含测试计数声明;NRIR48 关闭文档的 "996 passed, 37 skipped" 与当前
  `pytest tests --collect-only -q` 的 1033 collected(996+37)一致,旁证引用数字仍有效。
  changelog "未运行代码测试" 自述与 git 状态(docs-only)一致。

### 8. Changelog 自述:逐项核对

| 自述 | 复核 | 结果 |
|---|---|---|
| 14 个本地链接全部存在 | 脚本逐一验证 | PASS |
| G0—G8 九阶段、12 个 claim ID 可定位 | 9 阶段、12 ID 均确认 | PASS |
| git diff --check PASS(含 untracked 等价检查) | 独立重跑一致 | PASS |
| dol lint --soft ok=true, miss=[] | 独立重跑一致 | PASS |
| DocOps events ev005726/727/813/819 | 全部存在于 ev.jsonl | PASS |
| docs-only、未运行代码测试、无 GPU benchmark | git status 确认仅文档变更 | PASS |
| PR-12I/J/K/L、PR-13D、IR-5、NRIR48 数字逐项复核 | 见本报告第 5 节,1 处 minor | PASS-with-minor |
| 反方审计 approve-with-minor、5 项 minor 已全部修正 | **不可现场复核**(见 U-1) | NOT-AUDITABLE |

## Findings

### minor

- **M-1 | plan §6 历史表 PR-12J 行 |** 证据:`change_2026-07-14_pr12j_compile_amortization.md` 的
  compile phase 为 323.67/480.00/1299.12 ms(L69-71);"compile 约 0.29–1.48 s" 实际出自
  `change_2026-07-13_pr12ef_runtime_pareto_heldout.md:57`(PR-12EF 的 compile overhead 口径)。
  数字本身真实存在,但归到 "PR-12J compile amortization" 行会误导审计者到错误出处。
  建议:该行改为注明出处(如 "compile overhead 约 0.29–1.48 s(PR-12EF 口径);PR-12J compile
  phase 0.32–1.30 s"),或直接改用 PR-12J 自身数字。

### informational(不要求修改)

- **I-1 | plan §12 artifact/replay 合同 |** 合同文件清单(manifest/environment/queries/results_raw/
  normalized/summary/failure_rows 等)为前向预注册 schema,与当前仓库 artifact 实际结构
  (formal.json + manifest.json + shards/ + logs/)文件名不同。属 G0 待实现项,文档自身定位清晰,
  不构成失实;建议 G0 冻结时显式说明与现有惯例的映射关系。
- **I-2 | changelog "5 项 minor 已修正" |** 仓库中不存在该反方审计的独立记录文档,被审文档为
  untracked 无历史版本,无法 diff 验证"已修正"。建议将反方审计报告一并入库以便后续复核。

### 不可现场复核项

- **U-1 | 成稿反方审计(approve-with-minor、5 minor 已修正)**:无任何可核对的审计记录或文档旧版本。
  本次外部审计独立进行,结论(approve-with-minor,仅 M-1 一项 minor)与该自述不冲突。
- **U-2 | PR-12J v4 artifact 原始 CSV**:`artifacts/phase7a-pr12/` 当前仅存
  `kernel-foundation-20260713`,PR-12J 的 v4 artifact 目录不在当前工作树;本报告对 PR-12J 的复核
  基于其关闭文档(正式记录),raw 行级复核无法现场进行。
- **U-3 | GPU 相关声明的物理重放**:本会话 nvidia-smi 不可用(与文档自述一致),所有
  MEASURED-CURRENT 环境声明以 CLI/库查询复核,无法在 GPU 上重跑任何历史 workload。

## Claim-boundary 结论

无漂移。performance_claimed=false、IR-5 VALIDATED-NO-GO、ASPLOS-ready No-Go、fused coverage 0/394、
property 9/9 unknown、BoundConv 40x=USER-REPORTED 六条边界在被审文档与 claims map/关闭文档间完全一致;
核心立场声明(单主线收敛、17.1%/1.20x Amdahl、stream_id="sync"/trace-only event/logical arena、
auto_LiRPA 强 baseline、只做 G0/G1 + ≥20% opportunity gate)在文档中一致表达且经独立复核成立。
