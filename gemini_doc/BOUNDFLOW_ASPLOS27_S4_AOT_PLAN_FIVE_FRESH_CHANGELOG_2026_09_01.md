# BoundFlow S4 AOT PlanTemplate 与 five-fresh 性能闭环修改记录

date: 2026-09-01
stage: S4 AOT exact-call replacement
status: implemented-and-locally-validated
external-audit: not-requested
performance-claimed: false

## 1. 本轮解决的问题

上一版 S4 已经把 RVIR-v4 exact call 接到 prepared evaluator，但静态 prepare 仍依赖
`source_capture.pt`。这意味着它只能重放一个冻结实例，还不能证明同一份编译计划可以在新的
same-solver 进程里接收真实动态 α、β、split/history 和 incoming coefficient。

本轮把该依赖替换为可序列化的 AOT `PlanTemplate`，并完成 5 组、共 10 个独立进程的 B4-A/S4
交替对照。目标是先回答“通用准备与 exact-call 接入后，真实 query 到底快多少”，不把本轮包装成
外审或论文 claim。

## 2. 主要修改

### 2.1 Tensor-free AOT PlanTemplate

新增 `boundflow/runtime/asplos27_s4_exact_call_plan_template.py`：

- 离线绑定图拓扑、布局、shape/dtype、参数内容摘要、ReLU/α/β/split 语义和 capability；
- 模板本身不保存运行时 tensor；
- 支持 canonical JSON、稳定 hash、严格反序列化和 fail-closed 动态签名校验；
- 从模板实例化 persistent physical plan、seed tensor 与 16 个持久 buffer；
- 静态参数来源复制计数如实为 0，不再伪装为 query 期输入。

离线产物：

- `artifacts/asplos27-s4-exact-call-plan/resnet2b-prop0-v1/plan_template.json`
- `region_template_hash = e0019fa053f484ad61eaeca2792253baa395f980277de0c9bfb6a55dc9e5c0b0`
- `core_template_hash = e15a586c3f57ef9914a3931cb477bb6eb641f013bc89dfe1605ad4d0a7822cf5`

### 2.2 exact-call 动态重绑定

`asplos27_s4_exact_call_bridge.py` 增加从模板 prepare 的入口。warm prepared program 与 query
无关；每次 exact call 只接收并校验当前 solver snapshot/签名，再重绑定真实 α、β、split/history
和 incoming coefficient。receipt 新增 `region_template_hash`，并明确：

- `source_capture_runtime_dependency = false`
- `compile_inside_exact_call_count = 0`
- `provider_callback_count = 0`
- `fallback_count = 0`

### 2.3 公平的 post-prepare 环境窗口

静态 prepare 平均约 6.46 秒，会先加热 GPU。若直接从进程启动到 query 结束取环境窗口，candidate
会因为自己的 compile/warmup 被环境门禁拒绝，且 B4-A/S4 的窗口 scope 不一致。

因此 timing worker 新增 post-prepare 模式：双方都先完成静态准备，再同步、等待 GPU 回到冻结的
cool-idle 条件，重置 peak memory，然后才开始 query 环境窗口。没有降低温度、功耗或进程门禁。

### 2.4 five-fresh raw-first driver

新增 `run_asplos27_s4_same_solver_five_fresh.py`：

- 5 个 pair、B4-A/S4 交替先后顺序；
- 每个 worker 为独立 αβ-CROWN 进程；
- 环境未准入时保留原始 attempt 并重试，不覆盖失败证据；
- 离散语义 exact，lower 采用冻结的 `2e-4` 容差并要求 sign exact；
- manifest、protocol、summary 和逐 worker raw 可标准库 replay；
- artifact 中绑定当前 7 个代码/模板文件的 SHA256，现工作树逐项 MATCH。

## 3. five-fresh 结果

正式 artifact：

`artifacts/asplos27-s4-same-solver-aot-five-fresh/resnet2b-prop0-v1`

| 指标 | 结果 | 冻结门槛 | 判定 |
|---|---:|---:|---|
| pair / 独立 worker | 5 / 10 | 5 / 10 | PASS |
| 环境准入 | 10/10 | 10/10 | PASS |
| 离散求解语义 | 全部 exact | 全部 exact | PASS |
| lower 最大绝对误差 | `2.622604e-6` | `<=2e-4` | PASS |
| lower sign | 全部 exact | 全部 exact | PASS |
| S4/B4-A core 几何平均 | `1.115660996x` | `>=1.20x` | **未达到研究门槛** |
| S4/B4-A core 最差 pair | `1.001228544x` | 披露 | 未回退但无稳定余量 |
| S4/B4-A query 几何平均 | `1.116354455x` | parity `>=1.00x`; research `>=1.15x` | **通过 parity，未达到研究门槛** |
| S4/B4-A query 最差 pair | `0.989028582x` | 披露 | 一组轻微回退 |

5 组 query speedup 分别为：

`1.23862x, 1.13121x, 1.11063x, 0.98903x, 1.12656x`。

因此本轮可陈述的事实是：在该 ResNet2B/prop0 same-solver 样本上，AOT S4 相对旧 B4-A 的
完整 query 几何平均快约 `11.64%`；不能陈述稳定 `1.15x`，更不能外推到 complete verifier、其他
模型或“10x”。`performance_claimed` 继续保持 `false`。

## 4. 成本与剩余瓶颈

中位耗时：

| 项目 | B4-A | S4 |
|---|---:|---:|
| core | `249.484 ms` | `225.328 ms` |
| query | `1427.232 ms` | `1276.462 ms` |
| S4 setup | — | `5.208 ms` |
| S4 IR prepare/guard | — | `1.740 ms` |
| S4 optimizer | — | `49.901 ms` |
| S4 terminal handoff | — | `18.277 ms` |
| S4 static prepare | — | `6474.051 ms` |

静态准备不计入 warm query headline，但必须摊销。按 5 组平均 query 节省量计算，cold
break-even 为约 `42.68` 次 query。对复用不足的 workload，AOT/cache 命中是准入条件，不能忽略
这项成本。

最差 pair 中 S4 core 正常、query 却升到 `1439.416 ms`，说明当前主要不稳定项已经不是模板校验，
而是 core 外的 host/query residual。下一阶段应测量并拆分：

1. solver pre/post-core、KFSB/branch、queue/commit 与同步边界；
2. terminal export/handoff 是否仍有可移出 query 的 host 工作；
3. 10 次 evaluation 中的提交、版本检查和 Python 调度；
4. 静态模板/module 的磁盘与进程级 cache 复用，降低 6.46 秒 cold cost。

在 residual 归因前，不继续用新的审计轮次代替性能优化，也不靠放宽环境门禁得到 headline。

## 5. 验证

- five-fresh artifact replay：PASS；
- 模板/code revision SHA256：7/7 MATCH；
- exact-call/template 专项：`4 passed`；
- S4/FSG3/FSG4 相关回归：`279 passed`；
- 全量 pytest：`2098 passed, 4 skipped`；4 个 skip 均为已披露环境边界，零失败；
- Black：7 个触及文件 clean；mypy（`--follow-imports=skip`）7 个文件 clean；
- pylint：`10.00/10`；`git diff --check`：PASS；
- DocOps change/validation：`ev021284` / `ev021285`；
- 外部审计：本轮按用户要求不启动。
