---
status: final
updated: 2026-08-06T16:55:00Z
type: audit
topic: boundflow
slug: external-audit-nrir49a-g1-closure-2026-08-06
---

# 外部审计：NRIR49A G1 GPU Attribution VALIDATED-NO-GO Closure(2026-08-06)

## 后续范围纠正说明（不改审计原文）

下方审计正文作为当时的外部审计原文完整保留，其数据复核与 approve verdict 仍然有效。后续路线复审
进一步明确：该 `VALIDATED-NO-GO` 仅关闭 selected-CROWN-only incremental G2/G3，不关闭
BoundFlow operator→IR→JIT→runtime→memory 的累计全栈路线。正文中的约 `1.0764x` 只表示将实测
selected-CROWN 区域降为零耗时的 deletion-only Amdahl 上限，不是 BoundFlow 全栈上限；正文认可的
`gpu-winner-reselection` 是 closure 当时的历史 next route，当前已由
[Full-Stack GPU Baseline and Attribution v1](BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md)
取代。冻结 artifact、manifest、payload、hash 和审计正文均未因此改写。

- 审计对象:分支 `feat/nrir49-gpu-selected-crown-opportunity-v1`,HEAD `7836e11`
  "docs: close NRIR49A GPU attribution no-go"。
- 审计方法:不信任执行方任何汇总数字;从冻结 artifact 的 raw JSONL、manifest、系统日志与
  当前源码独立重算/重跑。
- 审计环境:Python `/home/lee/miniconda3/envs/boundflow/bin/python`(3.12.12),审计时本机
  `nvidia-smi` 可用。

## 总体 verdict:**APPROVE**

G1 以 `VALIDATED-NO-GO` 关闭成立。所有 headline 数字(7.0986% / 7.0523% / 1.353% /
Amdahl 上限 / 扰动比)均可从 raw 数据逐位独立复现;预注册门禁早于实验运行约 13 小时冻结,
无事后挪门槛;replay 为语义重算且通过;测试与工具链声明全部属实。仅有 4 项 minor 观察,
不影响结论。

## 逐项结论(8 项复核清单)

### 1. git / 推送状态 — 通过

- 分支 `feat/nrir49-gpu-selected-crown-opportunity-v1`,HEAD=`7836e110ba34bb009c4d80d31a642a1187be2b9b`;
  `git fetch` 后 `origin/feat/...` 同为 `7836e11`,本地与远端一致,已推送属实。
- closure 提交 `7836e11` 为纯文档提交(12 文件,+294/-27):`.docops`、change_log、
  08-05 计划(仅追加 8 行 FORMAL OUTCOME)、G1 plan/changelog、claims_map、执行备忘录、
  current_status、closure 变更记录。**不含任何代码或 artifact 改动**。
- 工作树仅 `M .docops/ev.jsonl`(30 行新增,为 closure 之后下一话题会话的 PostToolUse hook
  记录,时间戳 08:38–08:46Z,晚于 closure 提交 08:07Z);manifest 记录的
  `git_dirty_paths` 恰好也是 `M .docops/ev.jsonl`,与实验时工作树状态一致,非新污染。

### 2. Amdahl 逻辑与门槛来源 — 通过

- 独立验算:`1/(1-0.07098631834282758)=1.076410x`;`1/(1-0.070523288963519)=1.075874x`。
  "whole-queue Amdahl 上限约 1.076x" 表述准确。
- 反解可行性:`s_queue+1/1.2-1=-0.0957<0`、`s_complete+1/1.15-1=-0.0599<0`,两 scope 均
  INFEASIBLE,`r_queue_required/r_complete_required/r_latency_required=null` 正确。
- 门槛确为预注册:20% opportunity 门槛、Amdahl 反解公式、`1.20x/1.15x` 目标与
  `>10x/INFEASIBLE` kill 规则在 `38f4264`(2026-08-06 02:34 +0800,`git log -S` 确认)
  进入 08-05 计划文档,早于正式运行(15:25:57 CST,systemd journal)约 13 小时;
  `7836e11` 对该文档的 diff 仅为追加 FORMAL OUTCOME 8 行,门禁原文未动。**无事后挪门槛**。
- G8 latency headline(31-node queue geomean ≥1.20x 且 complete-query ≥1.15x)在 08-05
  §G8 与 §5.2.1 均有明文,closure 的比较口径忠实引用。

### 3. 原始数据独立重算 — 通过

审计脚本直接解析 `results_raw.jsonl`(62.5 MB,5 workers),不复用 normalized/summary:

- queue share(chunk 32,两 clause:`sum(selected_device_ns)/sum(queue_stream_ns)`):
  `[0.07065126095019457, 0.07146604699858099, 0.07108307584240295, 0.07088154217810674,
  0.07098631834282758]`,中位=`0.07098631834282758` —— 与声明 **逐位一致**;每行
  31 nodes/15 sibling groups/max depth 4 断言通过。
- complete share(`selected_all_device_ns/stream_ns`):中位=`0.070523288963519` ——
  **逐位一致**(另有 child-selected 变体中位 0.065450,未被混用)。
- paired profile/control wall ratio 中位:clause 2=`0.999304435327957`、clause 3=
  `1.0067470427656482`,均 ≤1.05 —— 逐位一致,instrumentation 可审计判定成立。
- fresh 证据:5 个 worker 各自由独立子进程执行(`_run_worker_subprocess` 每次新起
  `sys.executable`),raw 中各自 nvidia-smi 快照显示**互不相同的递增 PID**
  (15245/15549/16102/16437/16769)、环境 hash 各异、GPU 温度 41→60→62→61→59 °C
  呈物理合理的sequential负载曲线。systemd journal 独立证实
  `boundflow-nrir49a-g1-r3.service` 15:25:57 启动、wall 30m54.563s、内存峰值 2.1G、
  stdout 打印同一 summary/replay hash —— 与变更记录完全吻合。
- 无静默丢弃:`failure_rows.jsonl` 为 0 行(sha256=空文件 hash `e3b0c44…`);变更记录与
  changelog 显式披露了三次无 JSON 落盘的失败尝试(cgroup 回收 ×2+前台、整 queue CUPTI
  host OOM ×1)与 retry-2 bitwise hash 失败后按预注册浮点容差修正 parity 的历史,
  正式计数仅含 retry-3 的 5 个完整 worker。
- 离散度:queue share 相对极差 1.15%、complete 0.74%,五轮逐轮值已在 plan Formal Closure
  节完整列出 —— 离散度已报告且很小。

### 4. memory 数字 — 通过

- 从全部 profile/control/complete 行取 peak:max allocated=`81,854,976 B`、
  max reserved=`111,149,056 B`;物理总量 `8,214,937,600 B`(environment.json)。
  比例=`0.009964162844036697 / 0.0135301156761069` —— 与声明 **逐位一致**
  ("峰值占 1.353%" 取 reserved 口径,allocated 口径 0.996% 亦同时披露)。
- 预注册准入条款(08-05 §G1:自然/public workload 或合法 domain batch 达 80% 物理预算的
  `B80_alloc/B80_reserved`,或真实 `B_OOM`;禁止调低软件 budget 伪造)判定:峰值 ~1.35%
  ≪80%、无 OOM、合法 domain batch 上限 1 → admission 失败、G8 memory path=`N/A`。
  判定严格符合预注册条款,无降格解释。

### 5. artifact 完整性 — 通过

- replay(语义重算,非 summary 自洽):`replay_artifact()` 从 raw 逐行 `build_summary`
  重建并逐字段比对 stored summary、重算 normalized 全表、校验 manifest hash=
  canonical_hash(manifest 去 hash 字段)、code_revision==当前源码 sha256、8 个文件 digest、
  query 身份。审计独立执行 exit 0,stdout 与 `replay_stdout.txt` **逐字节一致**:
  `status=replay-passed`,decision=instrumentation PASS / queue opportunity FAIL /
  latency feasibility FAIL / memory admission FAIL / next_route=`gpu-winner-reselection`,
  summary hash=`7eefe6a7…ab50`。
- 独立 `sha256sum` 8 个文件,全部与 manifest `files` digest **逐位一致**;
  manifest hash=`d0272fe4…c81f` 与声明一致。
- manifest 绑定:git head=`c4fd0bb`(artifact 生成点,closure 文档提交的父提交,合理)、
  `git_dirty_paths`、VNN-COMP commit=`90419aad`(与 08-05 计划 frozen input 一致)、
  6 个代码文件 sha256(与当前磁盘文件**逐一相符**,见第 8 项)、worker_hashes 5 个全部
  与 raw 内 `worker_hash` 匹配、environment.json 含 GPU 型号/UUID/driver/clock/温度/功耗/
  compute PIDs。
- 环境证据真实性交叉验证:审计时本机 `nvidia-smi --query-gpu=uuid` 输出
  `GPU-0d3ee0a6-e7da-6b3b-69c4-2153bf99ae8f`,与 artifact 中 5+1 份快照**完全一致**;
  driver `610.43.03`、8188 MiB、`kwin_wayland` PID 1497 占 7 MiB 均与当前系统状态一致。
  GPU 环境证据真实。

### 6. 测试与工具链 — 通过

- 全量:`pytest tests -q -rs` → **1059 passed, 3 skipped**(422.06s),与声明一致。
  skip 原因经 `-rs` 核实:1× TVM 可用故跳过 no-TVM smoke(去重编译开销),
  2× frozen VNN-COMP checkout 当前不在磁盘(见 minor-3)。
- targeted 11:`tests/test_nrir49a_g1_gpu_attribution.py` 的 10 项合同测试 +
  `tests/test_phase7a_pr12k_cupti_profiler.py` 的 1 项既有 CUPTI 测试 =
  **11 passed**(3.41s),与 changelog "10 项+既有 CUPTI 共 11 passed" 表述精确对应。
- mypy:`mypy scripts/run_nrir49a_g1_gpu_attribution.py` → `Success: no issues found`。
- pylint:runner+测试文件 → **10.00/10**。
- `dol lint --soft` → `{"ok":true,...,"soft":true}` PASS。
- 说明:changelog 中段曾记 `1057 passed`(正式运行前),closure 声明 1059;当前 HEAD 实测
  1059,以 closure 为准,差异为期间新增测试所致,非矛盾。

### 7. claim 边界 — 通过,无漂移

- closure/plan/claims_map/current_status/执行备忘录五处口径一致:`VALIDATED-NO-GO` 仅否定
  selected-CROWN GPU 优化路线(G2/G3/TIR/JIT/融合 gated off),`performance_claimed=false`,
  不形成 speedup、竞品、multi-workload、solved verdict、memory headline 或 ASPLOS-ready
  claim;ASPLOS-ready 仍为 NO。
- 历史结论未被改写:fused coverage `0/394`、IR-5 Global p90 regret 1.26160x `VALIDATED-NO-GO`、
  NRIR48 CPU attribution `VALIDATED-REDUCED`、9/9 unknown 等在 08-05 文档"不能主张"清单
  原样保留(7836e11 仅追加,未改动)。
- "下一步重新归因 GPU whole-queue winner" 与 08-05 预注册 kill 路径一致
  (§5.2.1:"否则本 selected-CROWN 路线 NO-GO/重选 winner";§G1 门禁:"若未过…应按 GPU
  profile 重新选择 winner";roadmap 图:"bottleneck/latency 可达性不成立 → memory-only /
  重新选择 winner / 关闭路线")。memory-only 分支因 admission 失败而被正确排除。
- 执行备忘录与 current_status 中 "closure 文档变更尚待提交" 一句随 closure 提交入库后即
  自我过时 —— 措辞瑕疵,不构成事实错误(minor-1)。

### 8. 新增 profiling 代码质量 — 通过

- 只读观测:runner 为新增独立脚本;production 5 文件
  (`native_intermediate_refinement.py` 等)最后改动停留在 `351f5ce`(NRIR47,早于本分支),
  当前 sha256 与 manifest `code_revision` 逐一相符 —— production 源码、TIR、kernel、
  默认 chunk 均未修改,声明属实。
- observer on/off 等价:wrapper 仅在 worker 进程内 patch
  `refinement._run_selected_crown`;control 模式 `enabled=False` 为直通;语义等价由
  60/60 组离散结构 exact + raw 浮点逐叶 ≤2e-4(最大 abs/rel=
  `2.288818359375e-05/1.7107e-04`)双重门禁证明;33877 个数值派生 hash 差异显式计数
  而非误判。
- 插桩开销:几何/显存快照在 CUDA event 窗口外;queue 级扰动由 paired control 约束,
  实测中位 0.9993/1.0067 ≤1.05;CUPTI profile 明确排除在 timing summary 之外。
  计时口径干净。
- tamper 测试覆盖:同步重写 hash 篡改、profiler 源码篡改、cache 源码失配均有 fail-closed
  合同测试(10 项中的 3 项)。

## Findings

| severity | 位置 | 证据 | 建议 |
|---|---|---|---|
| minor | gemini_doc/asplos_execution_memo_v1_0.md:8、current_status_after_pr13.md:5 | "closure 文档变更尚待提交" 随 7836e11 入库后即过时 | 下一文档轮次顺手修正措辞;无需补救提交 |
| minor | artifact raw worker schema | worker 记录无显式 wall-clock 时间戳;freshness 依赖 PID/温度/journal/文件 mtime 间接证据 | 后续协议在 worker envelope 增加 start/end epoch 与 monotonic 读数 |
| minor | 环境可复现性 | 审计时 frozen VNN-COMP sparse checkout 不在磁盘(2 个无关测试 skip);artifact 仅绑定 model/property sha256,未含原始字节 | 在 artifact 或长期存储中固化 model/property 字节或可用镜像位置 |
| minor | gemini_doc/BOUNDFLOW_NRIR49A_G1_GPU_ATTRIBUTION_V1_CHANGELOG_2026_08_06.md:35 | 中段记录 `1057 passed` 与 closure 的 `1059 passed` 数字不同(期间新增测试) | 无需处理;当前 HEAD 实测 1059 与 closure 一致 |

无 blocker / major finding。

## 不可现场复核项

- **GPU 实验本身不重跑**(按审计委托约定):正式 5 worker 的物理执行无法在本审计中重演;
  但其真实性由 systemd journal(服务起止、wall 30m54s、2.1G 峰值、stdout hash)、5 份
  互异 nvidia-smi 快照(PID/温度/功耗)、与当前系统一致的 GPU UUID/driver/kwin PID
  三方独立证据交叉支撑,未见伪造迹象。
- 三次失败尝试(cgroup 回收、CUPTI host OOM)无落盘 JSON,只能依据执行方披露与
  `failure_rows.jsonl` 为空的一致性采信;披露本身完整且与 retry 历史自洽。
- `dol ch/va add` 记录未执行:受审计委托"除写审计报告外不改仓库"约束,DocOps 写入从略;
  `dol lint --soft` 只读校验已 PASS。

## 对下一步建议的评价

"重新归因 GPU whole-queue winner" 方向正确且是预注册路径的唯一剩余出口:selected-CROWN
在 GPU queue 仅占 7.10%,而 chunk sweep 显示 chunk 8 时占比可达 22.9%(sweep 仅作归因、
不得回写 production,边界已被正确声明),说明 GPU 时间主要不在 selected-CROWN region。
新归因应复用本轮的 read-only runner、paired-control 扰动门禁与 artifact/replay 合同,
并在协议中补 worker 时间戳(见 minor-2)。同意不启动 selected-CROWN TIR/JIT/融合;
G2/G3 gating 逻辑严格符合预注册。
