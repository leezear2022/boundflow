# FSG4/B3 36-Process 正式计时外部审计报告(Round 2)

- task: fsg4-b3-formal-timing-20260814
- round: 2(substantive)
- delivery: fsg4-b3-formal-timing-20260814/r002/delivery
- auditor: external-model(独立外审)
- date: 2026-08-15
- audit HEAD: 214e536(result commit d1c9505 的后续)
- artifact source: 36e9069ca4f21183c9b36d74024de0ca8b20f59c

## 总体 Verdict: APPROVE

同意 `VALIDATED-REDUCED-B3` 关闭;同意外审通过后仅开放 B4 cumulative candidate,
B5—B7 与最终 `1.20x queue / 1.15x complete-query` 门槛继续保持关闭。

所有结论均由审计方从冻结 raw(`worker_runs.jsonl`、`workers/run_*.json`、
`metadata/run_*.json`、`logs/*`)用独立脚本重算得出,未调用 executor 侧
`derive_fsg4_b3_timing_evidence()`,未采信 `summary.json` 任何数字。

## 审计方法

- 独立重算脚本:`/tmp/fsg4_b3_independent_audit.py`(纯标准库,只读 raw JSON/JSONL);
  最终输出 `ALL INDEPENDENT CHECKS PASSED`(44 项检查全 PASS)。
- manifest `files` 段 156 个文件逐一 sha256 复核,全部一致。
- `code_revision` 段 19 个源码文件的 sha256 与 source commit `36e9069` 的 git blob
  逐一比对,全部一致(覆盖 runner/replay/tamper 及 B3 全部 runtime 依赖)。
- 外部输入独立复核:αβ-CROWN `e5c7e17…`、auto_LiRPA `5a098e8…`、vnncomp2021
  `90419aa…` 三个外部 checkout 的 `git rev-parse HEAD` 与 protocol 记录逐一相同;
  `resnet_2b.onnx` 与 `prop_0_eps_0.008.vnnlib` 的 sha256 与 protocol 记录相同。

## AC1:Source、输入与正式协议 —— PASS

- `manifest.json`/`protocol.json` 的 `source_git_head` 均 exact 为
  `36e9069ca4f21183c9b36d74024de0ca8b20f59c`;`git cat-file -t` 确认该 commit 存在。
- code revision 19 文件全部与 source blob 哈希一致(见审计方法)。
- protocol 绑定前一阶段 five-fresh manifest:internal hash
  `457ab1adc8…1573` 与 file sha256 `bf8b3ecc…cb98`,
  审计方对 `artifacts/fsg4-b3-correctness-pairs/resnet2b-prop0-v1/manifest.json`
  独立重算,两者均一致。
- model/property sha256、αβ-CROWN/auto_LiRPA/VNN-COMP commit 均可从 raw 记录出发,
  与当前外部 checkout 独立核对一致(见审计方法);解释器(Python 3.11.15 /
  torch 2.11.0+cu130 / driver 610.57.04)在 36 个 worker envelope 中完全一致,
  GPU identity 单一(RTX 4060 Laptop, UUID `GPU-0d3ee0a6-…`)。
- 路径泄漏:`grep -rI '/home/' artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1/`
  命中 0;command projection 使用 `$VNNCOMP_ROOT`/`$ABCROWN_ROOT`/`$MODEL`/`$PROPERTY`
  占位符。raw solver 日志行尾空格属已知豁免,未发现其他本机路径。

## AC2:36 个 Fresh Worker 与固定顺序 —— PASS

- raw 36 行;按 (block_index, sequence_position) 排序后与 `protocol.expected_sequence`
  逐元素相同:六个 B0/B2/B3 全排列 block,每 block 每配置 1 control + 1 profile。
- 每配置 6 control + 6 profile(B0/B2/B3 各自核对)。
- 独立 subprocess 证据:`metadata/run_*.json` 各含 returncode=0、started/ended
  monotonic 区间,36 个区间串行且互不重叠;每 worker 独立 stdout/stderr 日志,
  日志头部均显示全新 αβ-CROWN 进程启动横幅与各自时间戳(如 run_00 21:26:58、
  run_17 21:32:10);runner 源码 `scripts/run_fsg4_b3_same_solver_experiment.py:361`
  使用 `subprocess.run` 逐 worker 拉起,非单 CUDA context 循环。
- raw-first/resume:同文件 `:652` 部分文件缺失即
  `FSG4/B3 partial worker cannot resume`;`:675` resume metadata(含 returncode、
  command projection)不一致即拒绝;`:693` resume protocol 与当前 source 不一致即
  拒绝;`_generate` 起始处 `_code_paths_clean()` 要求正式生成前 git 工作区干净。
  不存在从部分结果补写 formal manifest 的路径。

## AC3:Correctness 与物理 Activation —— PASS

- 独立语义对比(冻结 atol=rtol=2e-4、sign exact、离散字段 exact):
  每 block 内 B0/B2/B3 六个 worker 与 B0 control 参考逐项一致(30 对),
  且 B2-control vs B3-control 显式 6/6 一致;未读 summary 布尔值。
- B0 12/12 `replacement_mode=original_provider` 且 provider core 实际被调用;
  B2/B3 24/24 provider core/compute/update/fallback 全零。
- B3 12/12 具备 prepared template、PlanInstance、terminal Schedule、assembly、
  commit receipt、device commit audit 各一次,headline digest=0、
  timed candidate D2H=0,post-query audit 排除在计时外。
- B2 profile 6/6 counters snapshots/forward/D2H=`10/5/12`、optimizer=`10/9`;
  B3 profile 6/6 为 `0/4/0`、optimizer=`10/9`,另核对 template hit=1、
  module move=0、committed paths=12、KFSB=`3/3`。
- 18/18 control 行 `detailed_counts` 为 None(无详细 counter instrumentation)。

## AC4:Measurement Admission —— PASS

- 36/36 `environment.admitted=true`;runtime identity 去重计数=1;
  GPU (name, UUID) 去重计数=1。
- 18/18 profile closure 由审计方从 raw `profile_spans`(scope=core 的 wall 之和)
  对 `core_wall_ns` 独立重算:全部 < 0.01 门槛,最大值
  `0.002510499028552414`(≈0.0025104990,与 closure.json 逐字段一致,
  一致仅作为定义交叉确认,数值来源为独立重算)。
- profile/control query 扰动独立重算最大值:B0 `1.026924`、B2 `1.043622`、
  B3 `1.002677`,均 ≤1.05。
- headline ratio 只使用同 block control 配对(见 AC5 重算),profile 仅用于归因。

## AC5:独立 Ratio 与分类重算 —— PASS(重点项)

独立脚本直接从 `worker_runs.jsonl` 按同 block control pair 重算几何平均:

| 指标 | 独立重算值 | 冻结预期 | 结果 |
|---|---:|---:|---|
| B2/B3 core wall geomean | 1.0716174805930418 | 1.0716174805930418 | 完全一致 |
| B2/B3 query wall geomean | 1.0066228954759742 | 1.0066228954759742 | 完全一致 |
| B0/B3 query wall geomean | 0.9100012637918488 | 0.9100012637918488 | 完全一致 |
| 六个 B2/B3 core pair 最小值 | 1.0635877032562384 | 1.0635877032562384 | 完全一致 |
| B0/B3 core geomean | 0.5359654768687204 | (closure 0.535965x) | 一致 |
| B0/B3 peak allocated / reserved | 0.9986474155976598 / 1.0 | 无显存收益 | 一致 |

六个 core pair:1.090314 / 1.077497 / 1.063588 / 1.066105 / 1.064576 / 1.067877,
全部 > 1.05,最差 pair 未触发退化门禁。

按 plan 第 9 节冻结阈值独立分类:core geomean `1.0716` ∈ [1.05, 1.15),
query 相对 B2 不退化(≥1.0),但 B0/B3 query `0.9100` < 1.00 —— 恰好分类为
`VALIDATED-REDUCED-B3`;不满足 full `VALIDATED-B3`(core < 1.15 且未回 B0 parity),
也不满足 NO-GO(core ≥ 1.05 且 query 未退化)。peak allocated/reserved 无实质改善,
不产生任何 memory claim。

## AC6:Replay、Tamper 与冻结证据 —— PASS

- root replay(审计方实跑):
  `python scripts/run_fsg4_b3_same_solver_experiment.py replay --artifact-dir …`
  输出 `status=validated-reduced-b3`、36 runs、decision_inputs 与独立重算逐值一致,
  重建 `summary_hash=4c19afd4…99bac` 与 manifest 绑定值一致。
- tamper probe(审计方实跑,报告写 `/tmp/fsg4-b3-formal-audit-tamper.json`):
  10/10 rejected,全部 `manifest_file_digest_resigned=true` 且
  `manifest_hash_resigned=true`。其中:
  - `control-latency-outer-resign`:改 raw latency 并重签外层 digest →
    派生重算拒绝(`derived replay differs: paired_runs.jsonl`);
  - `b3-semantic-outer-resign`:改 raw semantic 并重签 → 派生重算拒绝;
  - `b3-profile-counter-outer-resign`:改 counter(forward 4→5)并重签 →
    冻结 counter 结构拒绝;
  - 其余 delete-worker / aggregate-order / b3-activation / b3-fallback /
    formal-preflight / protocol-sequence / summary-ratio 均被拒绝。
- 冻结 tamper report 文件 sha256 实测
  `bd392e5c…7e21b`,与 closure 记录一致。
- frozen test `tests/test_fsg4_b3_same_solver_artifact.py` 绑定 source
  (`SOURCE_GIT_HEAD` 常量)、固定 expected_sequence、direct activation 计数、
  protocol/manifest/summary/tamper 内部与文件 hash;单独运行 `6 passed`。
- targeted:`pytest -q tests/test_fsg3_same_solver*.py tests/test_fsg4_b3*.py`
  → `114 passed in 6.85s`。
- full:`pytest -q tests -rs` → `1314 passed, 3 skipped, 6 warnings in 464.30s`;
  skip 理由逐一核对:1 个 TVM 重复编译规避
  (`test_artifact_phase5d_smoke.py:118`)、2 个冻结 VNN-COMP checkout 不可用
  (`test_cross_axis_verification_batch_artifacts.py:70`、
  `test_root_projection_floor_artifacts.py:66`),与已知边界一致,不涉及本 artifact。

## AC7:Claim 边界与下游门禁 —— PASS

- `performance_claimed=false` 在 manifest、protocol、summary、closure.json、
  environment.json、36 个 worker row、36 条 activation receipt、18 条 paired row、
  tamper report 中逐一核对,全部为 false;manifest status=`validated-reduced-b3`。
- closure/handoff/plan 中无"快于 auto_LiRPA"或全栈 speedup 表述;唯一出现
  "headline speedup"处为禁止性语句("不得包装成 ASPLOS headline speedup")。
- 单 workload(ResNet2B property 0)、固定一次 solver prefix、单 RTX 4060 Laptop
  限制在 closure「Claim 边界与下一步」与 delivery known limitations 中保留。
- claims map(`gemini_doc/asplos_claims_map.md` 顶部 FSG4/B3 条目)与 closure、
  plan 第 23 节一致:`VALIDATED-REDUCED-B3`、外审待完成、B4—B7 关闭;无 claim
  漂移。plan 明确外审通过后只开放 B4 cumulative candidate,B5—B7 与最终
  `1.20x/1.15x` 门禁继续关闭。

## 流程核查

- r002 delivery 的 result_commit `d1c95059bb399b7cb01ce6a8b97f5149e21ae6de`
  真实存在(`git cat-file -t` = commit,message `test(bench): close FSG4 B3
  formal timing`),且是当前 HEAD `214e536` 的祖先;`git show --stat` 确认其包含
  delivery 所列全部变更文件;`git diff d1c9505..HEAD` 对 artifact、frozen test、
  closure 文档为空(证据自 result commit 起未被改动)。
- r001 F1(result_commit 元数据笔误)的 response 为 accept,r002 delivery 使用了
  正确的 full SHA,落实完毕。

## 不可现场复核项

- 36-process 正式运行本身(约 16 分钟 GPU 实验)按任务边界未原地重跑;本审计以
  冻结 raw 的重放 + 独立重算 + tamper probe 替代,符合 request 第 4 节已知边界。
- 正式运行时刻的温度/排他环境只能通过 raw 中 nvidia-smi 快照与 preflight 记录
  核对,无法回溯物理复现;36/36 admitted 与单一 runtime/GPU identity 已从 raw
  独立确认。
- αβ-CROWN/auto_LiRPA/VNN-COMP 三个外部 checkout 的当前 HEAD 与 protocol 记录
  一致,但外部仓库在运行时刻是否恰为该 commit 依赖 executor 的过程记录
  (commit 已绑定进 protocol_hash 与每 worker 的 protocol_identity)。

## Findings

无 blocker / major / minor。

### I1 [info] 解释器环境记录说明

- evidence: worker envelope `diagnostics.runtime_environment` 记录正式运行为
  Python 3.11.15 + torch 2.11.0+cu130(αβ-CROWN 侧 venv),与仓库开发 conda 环境
  (Python 3.12 / torch 2.12.1+cu132)不同;二者均在 raw 中明确记录且 36/36 一致。
- advice: 无需处理;后续阶段若更换 solver venv,protocol_identity 变化会自然
  fail closed。

### I2 [info] raw solver 日志行尾空格

- evidence: `logs/*.stdout.txt` 保留 αβ-CROWN 上游行尾空格,已被 manifest digest
  绑定;request 第 4 节明确豁免。
- advice: 无需处理;不要为格式检查改写日志。

## 结论

- 同意 `VALIDATED-REDUCED-B3` 关闭:是。
- 同意仅开放 B4 cumulative candidate(B5—B7 与 1.20x/1.15x 保持关闭):是。
