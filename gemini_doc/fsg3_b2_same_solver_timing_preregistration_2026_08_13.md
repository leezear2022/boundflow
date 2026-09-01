---
status: closed-baseline
updated: 2026-08-14T00:00:00+08:00
type: plan
topic: boundflow
slug: fsg3-b2-same-solver-timing
stage: s01
---

# FSG3 / B2 Same-Solver Timing 预注册

> **行政关闭注（2026-08-14）**：预注册正文和门禁未按结果回写；正式v5已按本文完成36/36 fresh
> GPU进程并以`VALIDATED-FSG3-B0-B1-B2-BASELINE`关闭。B2 query/core为
> `0.908400x/0.516767x`（B0/candidate），分类`MEASURED-B2-SLOWER`，不是speedup。正式证据见
> `gemini_doc/change_2026-08-14_fsg3_same_solver_formal_baseline.md`。

## 0. 冻结结论

本阶段只建立官方αβ-CROWN host solver内B0/B1/B2的可审计性能基线：

```text
B0 original provider
B1 RVIR typed passthrough -> original provider exactly once
B2 RVIR-v4 whole-core BoundFlow reference replacement -> provider/fallback zero
```

V4-3=`VALIDATED-WHOLE-CORE-REPLACEMENT`只提供计时资格，不提供性能结论。本预注册必须在读取任何
FSG3正式结果前提交；正式结果不得反向修改顺序、作用域、指标、门禁或排除规则。

B2是未优化reference executor基线，不是BoundFlow全栈终点。B2变慢只说明B3—B7需要回收哪些开销，
不得据此停止尚未实现的IR复用、fusion、JIT、runtime并行或arena/buffer reuse；只有correctness或
measurement auditability失败才阻止进入下一工程切片。

## 1. 固定身份

- preregistration base：`4edc621`；
- branch：`feat/rvir-v4-production-state-ownership-v1`；
- official αβ-CROWN：`e5c7e17bf0488843acb77b7519f59876717a49f4`；
- auto_LiRPA：`5a098e8f9fb5786a428a024981d833d303921f2d`；
- VNN-COMP benchmark：`90419aadcf06cf543ce5c1706cae1059dc9fa6cf`；
- workload：`cifar10_resnet:000` / ResNet2B property 0；
- model/property SHA256继续使用V4-3 formal manifest绑定值；
- device：本机RTX 4060 Laptop GPU，official competitor Python环境；
- seed=`100`，timeout=`60 s`，max_iterations=`1`，batch_size=`64`，auto-enlarge=`false`，
  alpha steps=`5`，beta steps=`10`，attack=`skip`，fresh isolated property copy。

FSG3只测该已完成whole-core correctness的固定workload。`mnistfc:2`及第二held-out family属于FSG5，
不得把没有whole-core replacement覆盖的workload伪装成B2。

## 2. 三个配置的物理定义

### B0 — `original_provider`

- 未修改的official `stage_solve.update_bounds_core`；
- replacement mode=`original_provider`；
- provider core恰一次，provider `compute_bounds`调用计数按真实执行记录；
- 仅安装与B1/B2共同的轻量outer/core event和post-measurement引用捕获。

### B1 — `rvir_passthrough`

- 从真实`pre_result/net/kwargs`构造并validate RVIR-v4 production typed snapshot；
- 记录snapshot identity、typed tensor/history数量与transport bytes；
- validation通过后恰一次调用原`update_bounds_core`，不得重试或旁路执行B2；
- replacement mode=`rvir_passthrough`，provider core恰一次，fallback为0；
- B1/B0量化adapter、typed state materialization与validation开销，不是BoundFlow计算收益。

### B2 — `whole_call`

- 使用V4-3关闭的production pre-state→10/9 optimizer mutation→native backward→KFSB→atomic
  copy-out→official post/queue完整路径；
- provider core/`compute_bounds`/`update_bounds`/fallback均为0；
- reference backend=`torch-eager-reference`，不得在本阶段暗中加入TIR、Inductor、CUDA Graph、
  multi-stream或arena优化；
- BoundIR/Plan允许物理驱动reference execution，但本阶段不复用跨process cache；B3才加入持久化
  graph/plan/schedule reuse；
- replacement mode=`whole_call`。

## 3. 冷、暖与计时边界

每个worker均为独立Python/CUDA进程。父进程记录`process_wall_ns`，但该值只诊断interpreter/loader
噪声，不进入主ratio。worker内CUDA同步后记录：

1. `cold_total_ns`：从B2 BoundFlow ONNX import/normalize/plan cold compile之前，到solver返回并完成
   最终CUDA同步；B0/B1无额外BoundFlow compile，因此等于各自query scope；
2. `boundflow_compile_ns`：只对B2记录一次raw ONNX→BoundIR/reference plan构造；B0/B1为0；
3. `query_wall_ns`：BoundFlow plan已处于process-hit状态后，完整official solver construction + verify；
4. `query_gpu_ns`：与query wall同边界的CUDA event elapsed；
5. `core_wall_ns`：exact `update_bounds_core`边界；B1包含typed transport，B2包含完整replacement；
6. `core_gpu_ns`：相同core边界的CUDA event elapsed；
7. `post_measurement_validation_ns`：计时闭合后才做GPU→CPU semantic payload复制，不进入上述scope；
8. peak allocated/reserved在cold scope前reset，按完整worker记录。

`cold_total_ns`是cold headline候选，`query_wall_ns`是process-hit/warm-plan候选；二者必须同时报告，
禁止只选更有利的一项。当前无disk cache，`disk_hit=N/A`必须显式写出。若B2 warm query更快，break-even：

```text
ceil(median(boundflow_compile_ns) /
     (median(B0 query_wall_ns) - median(B2 query_wall_ns)))
```

分母`<=0`时break-even=`not_reachable`，不得给伪有限值。

## 4. 正式顺序：36个fresh进程

正式运行六个block，覆盖B0/B1/B2全部六种排列。每个block中每个配置各有一个unprofiled control(`C`)
和一个profile(`P`)，二者相邻；mode顺序交替。总计`6 blocks × 3 configs × 2 modes = 36`个进程。

| block | config order | mode order | worker order |
|---|---|---|---|
| 0 | B0,B1,B2 | C,P | B0C,B0P,B1C,B1P,B2C,B2P |
| 1 | B0,B2,B1 | P,C | B0P,B0C,B2P,B2C,B1P,B1C |
| 2 | B1,B0,B2 | C,P | B1C,B1P,B0C,B0P,B2C,B2P |
| 3 | B1,B2,B0 | P,C | B1P,B1C,B2P,B2C,B0P,B0C |
| 4 | B2,B0,B1 | C,P | B2C,B2P,B0C,B0P,B1C,B1P |
| 5 | B2,B1,B0 | P,C | B2P,B2C,B1P,B1C,B0P,B0C |

因此每个配置有6个主control值、6个profile值；每个配置在三个位置各出现两次，任意配置对的先后顺序
均为3/3。B0→B1与B0→B2 ratio使用同block的control raw配对，不使用非配对全局均值。

## 5. Control 与 Profile

### Control

- 只允许outer/core各一对CUDA event、`perf_counter_ns`、peak memory counter与轻量引用保留；
- 禁止Python/CUPTI全调用trace、tensor CPU复制、digest或JSON写入落在计时scope内；
- 正式B0/B1/B2 latency ratio只使用control。

### Profile

- B0/B1记录official provider call count与phase；B1额外记录typed transport；
- B2记录`typed_pre_state / compile / optimizer / backward / kfsb / atomic_commit / official_post_queue`
  host与CUDA子区间；
- 每个子区间同时标注stack layer、solver phase、resource与cache state；
- profile wall只用于attribution与perturbation，不得替代control wall。

同block/config的`profile/control` paired wall中位ratio必须`<=1.05`。失败则latency verdict=
`NOT-AUDITABLE`，只能修instrumentation后整轮重跑，不得丢弃个别慢run。

## 6. 正确性与no-fallback门禁

计时结束后，每个run必须保存并由replay重算：

- status/success/visited domains、queue before/input/accepted/pruned/after、depth/history/threshold identity；
- final decision、split depth、batch size、n_verified/n_splits；
- target post lower全部finite；upper按§13 pre-run amendment使用finite payload + positive-infinity mask，
  B0/B1/B2同block mask/sign exact，有限值`atol=rtol=2e-4`；
- B0 vs B1离散结果exact；B0 vs B2离散结果exact；
- B1 typed request恰一次validate + provider core恰一次；
- B2 provider core/compute/update/fallback=`0/0/0/0`；
- source/config/model/property/device identity exact。

任一失败则该block与整轮FSG3 correctness=`FAIL-CLOSED`。不得因其耗时异常而删除run。

## 7. 环境与排他门禁

orchestrator在每个worker前后保存：

- driver、CUDA runtime、Torch、Python、GPU UUID/name/total memory；
- graphics/compute background process清单与显存；
- SM/memory clocks、temperature、power draw、performance state、thermal throttle reason；
- host load、CPU governor、AC/battery状态；
- worker exit code、timeout与完整stdout/stderr。

允许固定桌面compositor占用`<64 MiB`，但其PID/name必须跨run稳定；任何其它CUDA compute进程、
thermal slowdown、worker overlap、设备或电源状态变化均产生failure row并使正式轮无效。普通power-limit
状态只记录，不伪装为thermal failure。

## 8. 统计与结果标签

对`cold_total_ns/query_wall_ns/core_wall_ns/query_gpu_ns/core_gpu_ns/peak allocated/peak reserved`分别报告：

- 6个paired raw ratio；speedup定义固定为`B0 / candidate`，大于1才是更快；
- median、min/max、MAD、geometric mean；
- B1/B0 adapter overhead，以及B2/B0 cold、warm-plan、core三个边界；
- compile break-even或`not_reachable`；
- profile各层互斥critical-path closure与unclassified residual。

禁止只报告最佳run、把`gpu_sum`当wall、把B1 overhead从B2 headline偷偷扣除，或把correctness worker
历史stdout中的`func`时间当正式结果。

## 9. 状态机与后续门禁

- correctness、环境、profile perturbation、artifact replay全过：
  `VALIDATED-FSG3-B0-B1-B2-BASELINE`；
- B2 ratio `>1`只能记为`MEASURED-B2-FASTER`，ratio `<=1`记为`MEASURED-B2-SLOWER/NEUTRAL`；
- 两者都不是B7系统性能claim，`performance_claimed=false`保持到独立审计；
- FSG3可审计关闭后才准入B3；B2快慢不决定是否研究B3—B7；
- correctness失败：返回RVIR-v4修复，B3关闭；
- measurement/environment失败：修harness或整轮重跑，不形成速度结论；
- FSG4仍按B3 IR/Plan复用→B4 fusion→B5 JIT/CUDA Graph→B6 runtime→B7 memory顺序，最终系统
  gate只在B7 vs B0执行。

## 10. Artifact、Replay 与篡改探针

正式artifact至少包含：

```text
manifest.json
environment.json
worker_runs.jsonl
paired_runs.jsonl
profile_spans.jsonl
closure.json
summary.json
failure_rows.jsonl
replay_stdout.txt
README.md
```

manifest绑定本仓source commit/code paths、三外部仓commit、env lock、model/property/config、完整36-run
顺序及所有文件SHA256。replay只从worker raw重建配对、统计、correctness、no-fallback、perturbation、
closure和最终标签。

至少加入以下同步重签攻击：改control latency、删一run、交换config/mode顺序、改B1 provider count、改B2
fallback count、改semantic tensor、改temperature/thermal failure、改summary ratio。即使同步更新payload与
manifest digest，也必须被raw语义重算或冻结协议拒绝。

## 11. 实现切片

1. `FSG3-0`：本预注册、权威状态和DocOps关闭；
2. `FSG3-1`：typed timing/result schema、statistics/replay/tamper单测；
3. `FSG3-2`：B0/B1/B2 worker与轻量control计时；
4. `FSG3-3`：profile子区间、环境排他与36-process orchestrator；
5. `FSG3-4`：先运行非正式1-block smoke，只修合同，不改冻结正式门禁；
6. `FSG3-5`：从clean source commit生成正式artifact、replay、tamper、全量回归与外部审计交接。

## 12. 当前非主张

本预注册不声称B2更快，不声称BoundFlow已经实现TIR/fusion/JIT/multi-stream/arena，不声称固定
max_iterations=1等于31-node queue或complete-query headline，也不改变ASPLOS-ready=`NO`。它只把
下一次性能实验变成可证伪、可重放、不会事后挑指标的FSG3 baseline。

## 13. Pre-Run Amendment 1：Lower-Only Upper Sentinel

该修订发生在任何FSG3 real worker或正式timing结果生成之前。V4-3 formal evidence已明确本固定路径为
lower-only，provider与candidate的upper使用`+inf`作为“未请求”哨兵；原§6“lower/upper全部finite”与
已知production事实冲突。

修订后的canonical raw表示为：lower必须全部finite；upper保存同shape的finite数值数组与独立
`upper_positive_infinity_mask`，mask为true的位置数值固定编码为`0.0`。跨配置/mode比较要求mask exact，
只对mask=false的位置执行`2e-4` allclose与soundness检查。NaN、`-inf`、未掩码`+inf`继续fail closed。
该修订不改变计时顺序、指标或性能门禁，也不是观察结果后的调参。

## 14. Pre-Run Amendment 2：Profile Raw Spans 与 Core Closure

正式36-process artifact生成前，将timing schema从v1升级到v2。v1只携带closure/residual投影，无法由
第三方从raw span重算，因此v2要求每个profile run把有序、互斥、带monotonic offset的typed span纳入
canonical payload和stable hash。

closure分母冻结为exact `update_bounds_core`的`core_wall_ns`，分子为该配置所有`scope=core` span的
wall总和；compile与official post/queue分别保留为独立scope。布局固定为：B0=`provider_core/post`，
B1=`typed_pre_state/provider_core/post`，B2=`compile/typed_pre_state/optimizer/backward/kfsb/
atomic_commit/post`。删项、换序、重叠、wall投影或derived closure不一致均fail closed。

本修订是在非正式worker smoke发现证据可重放性缺口后、正式artifact之前完成；既有smoke不进入统计，
没有形成performance claim。详见
`gemini_doc/change_2026-08-13_fsg3_profile_schema_v2_prerun_amendment.md`。

## 15. Pre-Run Amendment 3：Formal Worker Cool/Idle Preflight（已由§17细化）

非正式block-0连续smoke与后台全量pytest重叠时，环境门禁正确观察到外部CUDA进程、58→64°C升温及
software thermal counter增长，0/6 run准入。为执行§7已经冻结的“无worker overlap、无thermal
slowdown”要求，正式orchestrator在每一个worker开始前执行相同的、非计时preflight：

- GPU温度`<=50°C`；software/hardware thermal reason均为inactive；
- AC供电；除固定`kwin_wayland <64 MiB`外无CUDA compute process；
- 每5秒检查一次，最长900秒；完整sample序列写入run metadata并由replay重算；
- preflight通过后恰一次执行该序列位置的worker。若worker内部仍出现thermal/overlap，则该run和整轮
  按原门禁失败；不得重试、删除或替换个别run。

preflight不在query/core计时scope内，不改变36-run顺序、指标、重复数或阈值。它固定fresh-process
latency实验的起始环境，不主张持续吞吐性能。

## 16. Pre-Run Amendment 4：Post-Initialization Worker Preflight（已由§17/§19细化）

formal v1执行7个连续位置后整轮中止：outer preflight在`<=50°C`放行，但fresh worker完成
Python/Torch/solver与CUDA初始化后，worker `environment_before`已约52°C，query内software thermal
counter增长，7/7均不准入。v1没有manifest或performance claim，完整证据与原因见
`gemini_doc/change_2026-08-13_fsg3_formal_v1_environment_abort.md`。

v2保留outer preflight以排除并发，并在每个fresh worker完成import/CUDA初始化后、任何compile/query
计时前增加第二道冻结检查：温度`<=45°C`、thermal inactive、AC，除worker自身和固定compositor外无
CUDA compute process；5秒轮询、900秒上限。完整sample进入worker envelope，replay重算最后一条准入
状态。若随后timed query的thermal counter仍增长，该位置和整轮仍按原门禁失败，不得个别重跑。

v2必须从36-run position 0完整重启，不得混入v1的7个值。orchestrator每个worker后立即刷新raw run与
metadata，以便任何整轮fail-fast仍保留环境证据。本修正改变检查位置，不改变计时scope、顺序、指标、
重复数或统计阈值。

阈值冻结依据：非正式单worker在48→50°C时thermal counter仍增加约1.35秒并被拒绝，而更早独立
B0 profile在45→46°C曾准入。该校准发生在v2正式36-run之前，pilot值不进入统计。

## 17. Pre-Run Amendment 5：独立温控与功耗耦合遥测（Schema v3，温度阈值已由§19取代）

本修订发生在任何可准入的正式36-run之前。v2尚未启动完整正式轮；一次48°C pilot与关闭
`nvidia-powerd`的诊断仍被旧门禁拒绝。进一步读取同一`nvidia-smi -q -x`快照并做受控CUDA矩阵乘后，
确认本机RTX 4060 Laptop、driver 610.57.04存在可重复的遥测耦合：software power-cap与software
thermal的reason同步，两个累计counter在负载前后逐值严格相同；硬件thermal counter始终为0。单进程
v3 pilot的两个软件counter同为`50,495 -> 168,411 us`，而GPU为45→46°C、T.Limit margin为42→41°C。

NVIDIA官方文档把SW power cap和SW thermal定义为不同clock-event reason，也明确T.Limit是距最大运行
温度的**margin**，不是绝对温度。因此不能把本机XML中的`gpu_temp_tlimit=42 C`解释为42°C温控阈值，
也不能在两个来源完全镜像时把单独的software thermal位当成独立物理证据：

- <https://docs.nvidia.com/deploy/nvidia-smi/index.html>
- <https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-api/dcgm-api-field-constants.html>
- <https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html>

正式schema升级为v3，并冻结以下fail-closed规则：

1. 每个preflight与timed interval均原样保存SW power、SW thermal、HW thermal的reason和累计counter，
   同时保存GPU温度、T.Limit margin与target temperature；
2. 只有在SW power与SW thermal的before/after reason逐点相同，且两个累计counter在before/after都
   逐值相等时，才将software thermal投影为driver-coupled power-cap alias；普通power limiting允许；
3. 任一不相等的软件thermal信号、任一HW thermal reason/counter增长、counter倒退、外部CUDA进程、
   设备身份漂移或非AC供电仍拒绝；温度起点仍保留outer `<=50°C`、post-init `<=45°C`；
4. worker PID进入canonical preflight，artifact replay从raw snapshot/process列表重新计算preflight与
   interval gate；只改`admitted`、coupled布尔值或摘要不能通过；
5. schema v3从position 0完整启动，v1的7个值及所有v2/v3 pilot一律不进入正式统计。

修订后的非正式block-0 smoke覆盖B0/B1/B2 × control/profile六路，6/6环境准入、语义失败0、所有profile
core closure通过。它只证明合同可执行，不形成性能主张。完整诊断见
`gemini_doc/change_2026-08-13_fsg3_coupled_power_thermal_telemetry.md`。

## 18. Pre-Run Amendment 6：父子 Timeout 一致性

首个schema-v3正式尝试完成32/36个位置且32/32环境准入；第33个fresh worker在连续负载热浸后，
post-init温度约55°C，因此按§16继续等待45°C门禁。worker允许等待900秒，但orchestrator仍沿用早期
`180s`子进程总timeout，在worker尚处于合法preflight等待时将其终止。该轮没有manifest、summary或
performance claim，重命名为
`artifacts/fsg3-same-solver-timing/resnet2b-prop0-v3-aborted-parent-timeout-32-of-36/`；32个latency不得
续跑、补齐或进入统计。

下一正式尝试只修正合同冲突，不修改温度、顺序、指标、样本量或性能门禁：父进程timeout固定为
`worker preflight 900s + 180s execution margin = 1080s`并写入manifest contract；若仍超时，必须持久化
partial stdout/stderr与`failed_worker.json`后fail closed。新轮从position 0完整重启。该修订发生在任何
完整、可回放的正式artifact形成前，详见
`gemini_doc/change_2026-08-14_fsg3_formal_v3_parent_timeout_abort.md`。

## 19. Pre-Run Amendment 7：Post-Init 温度门禁可达性（Schema v4）

父timeout修正后的v4 attempt从position 0开始，首个worker完整执行176次、约900秒post-init采样：温度
范围47–54°C，后半程长期在47–49°C回摆，GPU利用率0%、功耗约3.4–3.6W，所有样本的
`independent_thermal_active=false`，但从未达到45°C。worker最终按自己的900秒门禁退出；该attempt为
0/36、无run、无manifest/summary/performance claim，保留于
`artifacts/fsg3-same-solver-timing/resnet2b-prop0-v4-aborted-post-init-45c-infeasible/`。

这证明45°C低于本机“模型和CUDA context已初始化、无计算负载”状态的可持续温度，而不是更严格的有效
thermal-quality gate。NVIDIA raw T.Limit margin同期仍约38–40°C，HW thermal为0，且严格耦合之外的
SW thermal为false；继续要求45°C会使实验不可运行，却不增加对真实热降频的排除能力。

下一attempt在任何正式timing产生前冻结如下修订：

1. timing/artifact schema升级为v4；post-init绝对温度上限由45°C改为50°C，与outer上限一致；
2. independent SW/HW thermal、raw counter、T.Limit margin、AC、外部进程和device identity门禁保持不变；
3. 每个worker仍fresh、计时前等待、5秒轮询、900秒上限；50°C是inclusive，51°C继续拒绝；
4. 36-run全排列、control/profile、指标、统计、correctness、no-fallback和performance门禁完全不变；
5. 非零worker退出与parent timeout均必须产生`failed_worker.json`和partial logs；
6. 新attempt命名`resnet2b-prop0-v5`，从position 0完整执行，不得混入v1/v3/v4或pilot值。

该修订依据的是preflight feasibility raw，不是任何headline latency或候选速度；完整记录见
`gemini_doc/change_2026-08-14_fsg3_post_init_temperature_feasibility.md`。
