---
status: validated-no-go
updated: 2026-08-06T07:59:16Z
type: plan
topic: boundflow
slug: nrir49a-g1-gpu-attribution-v1
stage: s01
---

# BoundFlow NRIR49A G1 GPU Attribution v1 Plan

## Goal

在不修改 production TIR、kernel、默认 chunk、Planner policy、数学或 termination 的前提下，回答三个
可证伪问题：

1. fixed ResNet2B clauses 2/3 的 selected-CROWN 在 RTX 4060 Laptop GPU 上是否仍占 queue
   critical path 的至少 20%；
2. 从 GPU 实测 share 反解出的 queue `1.20x` 与 complete-scope `1.15x` 所需 region speedup 是否
   同时可达且不超过 `10x`；
3. 8 GiB 目标卡上是否存在自然、semantic-valid 的 physical-memory admission，还是 memory headline
   应预先标为 `N/A`。

G1 只形成 attribution 与 G2/G3 的 go/no-go，不形成 speedup、TIR、JIT、多流、arena、planner 或
ASPLOS-ready claim。

## Frozen Inputs

- branch：`feat/nrir49-gpu-selected-crown-opportunity-v1`；G0 closure=`fb97715`；
- device：NVIDIA GeForce RTX 4060 Laptop GPU，compute capability `8.9`，8188 MiB；
- BoundFlow：Python `3.12.12`、Torch `2.12.1+cu132`、TVM `0.23.dev0`；
- VNN-COMP：`stanleybak/vnncomp2021@90419aadcf06cf543ce5c1706cae1059dc9fa6cf`；
- model：`cifar10_resnet/onnx/resnet_2b.onnx`，SHA256
  `791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d`；
- property：`prop_0_eps_0.008.vnnlib`，SHA256
  `89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff`；
- clauses：original ordinals `2/3`；每条 queue=`31` nodes、`15` sibling groups、max depth `4`；
- target selection、split/branch、optimizer、node/depth、deadline、numeric policy和 termination全部冻结；
- production `backward_chunk_size=32` 不改；`8/16/32/64/128` 只由 profiler wrapper 覆盖并写入
  evidence，Plan 本身仍验证为原 production Plan。

历史 CPU 数字只作 lineage：NRIR48 selected-CROWN 占 child execute约 `71.77%/72.73%`，不得代替本轮
GPU share。

## GPU Queue Entry

完整 ranking floor 直接搬到 CUDA 的非正式探针在 60 秒 deadline 后触发既有
`parametric complete verifier cache coverage differs`；因此它不是 G1 profiling 入口，也不能被解释为
selected-CROWN 结果。

正式入口按原 floor 的数学构造在 GPU 上重建：

```text
shared top-width refinement (128, chunk 32)
  -> clause-specific objective-influence refinement (128, chunk 32)
  -> compile frozen objective-branch shared Plan
  -> execute NRIR45 prepared 31-node production queue
```

该入口的 admission probe 已保持 clause 2 的 `31` nodes、`15` groups 与历史 worst lower
`-35.53092575073242` exact；它不改变目标 ledger，只跳过与 G1 scope 无关且在 GPU 上超时的九子句
ranking floor。

## Measurement Protocol

### Repeats and order

- 正式 timing 至少 `5` 个 fresh worker processes；每个 worker只使用 GPU 0；
- chunk Latin order冻结为五个循环位移：

```text
r0: 8,16,128,32,64
r1: 16,128,32,64,8
r2: 128,32,64,8,16
r3: 32,64,8,16,128
r4: 64,8,16,128,32
```

- 每个 worker先完成 untimed import/ONNX lowering；root source构建计入 complete scope，但不计入
  queue scope；
- default chunk 32 另做 paired lightweight-control/profile perturbation，顺序按 repeat parity反平衡；
- 所有计时前后 CUDA synchronize；正式 device time用同一 current stream 上的 CUDA events；wall time用
  `perf_counter_ns`；
- 不锁频时记录每轮 clock、temperature、power、driver与其它 compute PIDs；唯一环境豁免是本机
  `kwin_wayland<=64 MiB` 的有界桌面合成器（当前7 MiB且完整记录），其余非本 worker PID一律
  fail closed，不进入 summary；该豁免不适用于Python/CUDA训练或推理进程。
- worker允许写入可恢复cache以避免中断后重跑；cache必须绑定profiler及五个production依赖文件的
  SHA256并重新走完整semantic validation，任一源码变化即拒绝旧cache并重跑fresh process。

### Per selected-CROWN call

每个 root/child call记录：

- scope、clause、node/call ordinal、requested production chunk与effective harness chunk；
- 每个 ReLU的 target indices/count、ragged segments、chunk count、objective shape；
- dense one-hot、index、lower/upper output的理论 bytes；
- CUDA event device time、host wall time、allocated/reserved before/after；
- module op sequence、input/output value、consumer/fanout、Conv/Linear signature、join/layout boundary。

### Queue and complete scope

每条 queue记录：

- 31-node/15-group/depth/branch/state/ancestry/target ledger与 worst lower；
- selected-CROWN event sum、queue stream critical-path event、queue synchronized wall；
- baseline allocated/reserved、peak allocated/reserved、物理总显存与 peak ratio；
- child selected share=`selected child event sum / queue stream critical path`；
- lightweight profiler/control queue wall ratio，用于限制插桩扰动。

default chunk 32 的 complete scope覆盖两次 root构建与 clauses 2/3 两条 queue，记录 all-selected与
child-selected两种 share；用于反解 headline complete-scope门槛。单独的代表性 CUPTI/PyTorch profiler
只重放一个来自已计时default32 queue的真实clause-2 child selected-CROWN call，负责 kernel count、CUDA
runtime launch API/self CPU time、memory event与同步 API来源，明确排除在timing summary之外；禁止profile
整条31-node queue，避免事件保留量改变宿主内存可行性。

## Correctness Gates

- 每个 chunk/repeat/clause均须为 31 nodes、15 groups、max depth 4；
- branch、state、ancestry、split ledger与 target ledger相对同 worker chunk32 exact；
- lower/upper finite且 conservative；worst lower相对 chunk32 `atol<=2e-4, rtol<=2e-4`；
- selected target覆盖、output ordinal、lower/upper shape/dtype/device exact；
- profile/control与chunk parity采用双门禁：所有离散结构（node/parent/branch/split/target/order/shape/dtype/
  device/policy）exact；所有raw浮点逐叶finite且使用同一冻结`atol=rtol=2e-4`。alpha/beta/bounds/
  score等浮点派生hash允许随合格ULP差异改变，但完整raw payload hash必须绑定artifact，replay须重算结构投影、
  逐叶数值parity及派生hash差异计数；这不是bitwise-float exact，也不改变数值门槛；
- 任一失败行进入 `failure_rows.jsonl`，不得汇入性能统计；
- instrumentation `median(profile/control queue wall) <=1.05`，否则 G1 timing verdict为
  `NOT-AUDITABLE-PROFILER-PERTURBATION`。

## Opportunity and Amdahl Gates

对每个 scope冻结：

```text
projected_scope_speedup(s, r) = 1 / ((1 - s) + s / r)
required_region_speedup(s, T) = s / (s + 1 / T - 1)
```

- `s_queue`：两条 queue selected child event sum / 两条 queue stream critical-path sum；
- `s_complete`：default complete scope all-selected event sum / complete stream critical path；
- `r_queue_required=required_region_speedup(s_queue,1.20)`；
- `r_complete_required=required_region_speedup(s_complete,1.15)`；
- `r_latency_required=max(r_queue_required,r_complete_required)`；
- 任一分母 `<=0` 为 `INFEASIBLE`；
- `s_queue<0.20`：selected-CROWN不是 GPU winner，停止 G2/G3并重新归因；
- `r_latency_required>10` 或任一 scope infeasible：G3 latency路线 `NO-GO`；
- 只有 `s_queue>=0.20`、两scope可行且 `r_latency_required<=10` 才准入 G2 qualification。

chunk sweep只做归因：最优 chunk不得回写 production默认配置，也不形成 speedup claim。

## Physical-Memory Admission

- 记录每个合法 chunk/queue的 allocated/reserved baseline与peak；
- 当前 selected-CROWN target selector要求 source batch=`1`，因此在没有新增语义前
  `maximum_semantic_valid_domain_batch=1`；不得用复制输入伪造 domain batch；
- `B80_alloc/B80_reserved` 仅在自然/public workload或已有合法 domain batch达到物理总显存80%时存在；
- `B_OOM` 仅记录真实 allocator OOM；任意调低软件budget不算 physical admission；
- 若所有合法运行 peak均低于80%、无真实 OOM且最大合法 batch为1，则 RTX 4060的 G8 memory path=`N/A`，
  不进入 G5 headline。

## Artifact and Replay

正式 artifact目录：

```text
artifacts/nrir49a-g1-gpu-attribution/
  resnet2b-prop0-clauses2-3-rtx4060-five-repeat-v1/
    manifest.json
    environment.json
    queries.jsonl
    results_raw.jsonl
    normalized.jsonl
    summary.json
    replay_stdout.txt
    failure_rows.jsonl
    README.md
```

manifest绑定 git/diff、submodule、model/property、environment、order、sync、gate版本及所有文件
SHA256。replay从 raw逐行重算 call→queue→complete统计、Amdahl、memory verdict和summary；即使同步重写
payload与manifest digest，修改 share、target ledger、node count或decision仍须被语义重算拒绝。

## Tasks

1. 实现 schema、lightweight CUDA-event profiler、static topology inventory与GPU root/queue entry；
2. 实现 fresh worker、Latin order、paired perturbation、memory与environment采集；
3. 实现 aggregation、Amdahl/memory verdict、artifact/replay与tamper tests；
4. 运行五个正式 workers及独立 representative CUPTI profile；
5. 根据预注册门禁关闭 G1，决定 G2 qualification、memory-only或路线 NO-GO。

## Validation

- unit：公式边界、Latin coverage、call/queue closure、semantic mismatch、memory N/A与tamper；
- targeted GPU：clauses 2/3、chunk32、31/31 nodes、artifact replay；
- full `pytest tests`；Black、mypy、Pylint；`git diff --check`；
- DocOps performance规则：正式结果至少5 repeats；unit/single probe不支持性能 claim。

## Formal Closure

2026-08-06 15:25—15:56 CST 在 RTX 4060 Laptop 8 GiB 上完成五个 fresh worker；Latin chunk顺序、
clauses 2/3、31 nodes/15 sibling groups/depth 4 与 default chunk 32 均按冻结合同执行。正式服务 exit 0，
五个 worker 全部通过 envelope、源码修订、结构语义、逐叶 finite/tolerance 与 worker hash 门禁；
`failure_rows.jsonl` 为0行。代表性真实 child selected-CROWN 调用的 CUPTI 证据包含5954个kernel、
5486次runtime launch、398次同步与5364个memory event，且不进入 timing summary。

五轮 queue share=`[0.0706513,0.0714660,0.0710831,0.0708815,0.0709863]`，中位数=
`0.0709863183`；complete share中位数=`0.0705232890`。两条 clause 的 paired profile/control wall
ratio中位数=`0.999304/1.006747 <=1.05`，故 attribution可审计。60组结构exact；raw GPU数值最大
absolute/relative diff=`2.288818359375e-05/1.710717646052519e-04`，均低于预注册`2e-4`，并保留
33877个数值派生hash差异与完整raw payload hash。

`s_queue=7.10% <20%`，selected-CROWN不是GPU winner。queue `1.20x`和complete `1.15x`目标均超过
对应Amdahl无限加速上限，因此`r_queue_required/r_complete_required/r_latency_required=null`，
latency路线不可达。最大allocated/reserved仅占物理显存`0.996%/1.353%`，最大合法domain batch=1、
无OOM，故physical-memory admission失败且该硬件的G8 memory path=`N/A`。按预注册规则，G1以
`VALIDATED-NO-GO`关闭，停止selected-CROWN G2/G3，不启动TIR实现，下一步重新做GPU winner归因。

正式 artifact 位于
`artifacts/nrir49a-g1-gpu-attribution/resnet2b-prop0-clauses2-3-rtx4060-five-repeat-v1/`；
summary hash=`7eefe6a716fa57874420bcda64487ad02578dae5926fc99426eaaef37d35ab50`，manifest hash=
`d0272fe431d68ba93ef17a69fe7bf9b7ef71c7ab02041680720e65fadd86c81f`。独立 replay exit 0、stdout
逐字一致，8个payload SHA256与manifest hash均独立重算吻合。`performance_claimed=false`；本结论
不是speedup、竞品比较、多workload、solved verdict、memory headline或ASPLOS-ready claim。

## Rollback

- 本阶段只新增 profiler/test/docs/artifact，不修改 production runtime；
- 删除 G1 runner/test/docs即可回退；外部 VNN-COMP sparse checkout只增加 benchmark文件，不改其commit；
- 若 formal run失败，保留 failure artifact并把 verdict降级，不回退或调门槛适配结果。

## Links

- changelog: [NRIR49A G1 changelog](BOUNDFLOW_NRIR49A_G1_GPU_ATTRIBUTION_V1_CHANGELOG_2026_08_06.md)
- G0: [NRIR49 G0 admission](BOUNDFLOW_NRIR49_G0_GPU_OPPORTUNITY_ADMISSION_V1_PLAN_2026_08_06.md)
- roadmap: [GPU compiler acceleration research v1.1](BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md)
- CPU lineage: [NRIR48 attribution](change_2026-08-05_nrir48_execution_cost_attribution.md)
