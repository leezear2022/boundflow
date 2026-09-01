# BAB4 + root-CROWN 累计接通与 prepared-owner 复用修改记录

status: validated-three-fresh-diagnostic
date: 2026-09-01
external-audit-requested: false
performance-claimed: false

## 1. 起点与目标

BAB4-GC 五对正式结果为 complete-query `1.0630537637x`、activation-BaB core
`1.1805908944x`。进一步从原始事件重算发现，约 `573 ms` 的查询中，root/incomplete verifier
约 `236 ms`，占约 `41%`；只优化 activation-BaB 不可能把局部收益完整传播到查询。

仓库此前已经完成 root-CROWN 的 terminal、普通 residual、projection residual 和 input-domain
流式 concretization。这些模块各自包含 TVM/TIR forward、custom backward、压缩 α 与 persistent arena，
因此本轮没有重写 kernel，而是把既有 root owner 与 BAB4-GC 组合成同一个 same-solver candidate。

## 2. 真实累计路径

新增配置：

```text
B4-A-GC control
  = prepared request + native root warmup + native root + BAB4 control + GC isolation

BAB4-GC-ROOT candidate
  = prepared request
  + exact root-TIR warmup
  + 同一个 prepared root owner 复用
  + compiled root
  + BAB4 four-segment core
  + GC isolation
```

相关文件：

- `scripts/run_bab4_root_gc_worker.py`：单进程真实 solver worker；
- `scripts/run_bab4_root_gc_three_fresh.py`：三组交替 fresh 对照；
- `boundflow/runtime/root_crown_full_pipeline_tir.py`：四段 prepared executor 的 warm-reset；
- 三个 `root_crown_*_live.py`：保留静态 admission、清空 query-local 事务与计数；
- `tests/test_bab4_root_gc_integration.py`：顺序、对称性和诊断边界合同。

## 3. 第一次失败与物理归因

最初候选在 warmup 后重新构造第二套 root executor。三组诊断为：

- query `1.0623475x`；
- root `0.9648443x`，即 root 反而慢约 `3.6%`；
- core `1.2069697x`。

为避免猜测，本轮加入 opt-in 分段归因。它同时记录 host submission 与 current-stream CUDA event，
默认关闭并强制 `included_in_performance_claim=false`。一次真实候选中：

- terminal/residual/projection/input-domain 的 5 次 forward + 4 次 backward CUDA 总量约
  `7.9 ms`；
- root 总时间约 `257 ms`；
- 优化器第一次 `compute_bounds` 为 `44.8 ms`，后四次只有约 `21.8–23.2 ms`；
- suffix/projection/input 三个 `_admit_static` 的首轮 host 总量约 `14.4 ms`，后续调用近似为零。

所以倒退不是 TIR kernel 慢，而是 measured query 更换 executor 后重复执行 GPU→host 的 α 坐标、
ReLU 静态状态与 input-domain admission，同步尖峰落入第一次 `compute_bounds`。

## 4. 修复：exact warmup 后复用 prepared owner

修复不再创建第二套 query executor，而是在 exact model/property warmup 完成后复用同一套 compiled
module、persistent arena、DLPack views 和静态 admission。新增 fail-closed reset：

1. warm transaction 必须完整出现 5 forward / 4 backward；
2. 四个模块必须 `fallback=0`，所有 staged transaction 必须为空；
3. 三个 bridge 必须已经完成静态 admission，并持有完整 cached ReLU/input state；
4. 只清空 launch、pointer、bridge activation 与 query-local pending counters；
5. compile identity、arena、views、静态 admission 不清空；
6. reset 只能发生一次，query receipt 记录 `exact_warmup_reuse_count=1`。

reset 本身约 `0.014 ms`，在 query 计时前完成。修复后诊断中：

- 三类 `_admit_static` host 总量由约 `14.4 ms` 降到约 `0.12 ms`；
- 第一次 optimizer `compute_bounds` 降到 `21.54 ms`，不再高于后四次；
- 单次 root 约 `232.56 ms`，相比修复前的约 `244–257 ms` 明显下降。

## 5. 三组 fresh same-solver 结果

原始目录：`/tmp/bab4-root-reuse-three-v1`。该目录是本机诊断数据，不作为已冻结 artifact 或论文
claim；本节只记录本轮工程决策证据。

| scope | geomean | worst pair | pair values |
|---|---:|---:|---|
| complete query | `1.0859868343x` | `1.0814562651x` | `1.09119 / 1.08534 / 1.08146` |
| root incomplete | `1.0321000398x` | `1.0203539988x` | `1.04425 / 1.02035 / 1.03184` |
| activation-BaB core | `1.1898317081x` | `1.1845423755x` | `1.18454 / 1.19524 / 1.18974` |

正确性与资源：

- 三组 discrete semantics 全 exact；
- lower 最大绝对误差 `1.6093254089e-6`，sign 全 exact；
- peak allocated ratio `1.0125873x`；
- peak reserved ratio `1.0153846x`；
- query `1.15x` 研究门槛仍未通过；
- `performance_claimed=false` 保持。

相对 BAB4-GC 正式 `1.06305x`，本轮 complete-query 提升到诊断级 `1.08599x`，增加约 2.3 个
百分点。root 从第一次累计接入的 `0.96484x` 翻转为 `1.03210x`。

## 6. 剩余瓶颈与下一工程动作

fresh pair 的 root 事件显示：

- compiled custom backward 比 native 累计少约 `20–24 ms`；
- 但完整 root 只净省约 `4.8–10.2 ms`；
- 五次 `compute_bounds` 仍累计约 `333–340 ms`（其中包含嵌套调用），exclusive host traversal
  与 native 基本相当；
- 四段 TIR kernel 已经不是主要分母。

因此下一刀不是再调单 kernel，而是建立 root `compute_bounds` 高层 transaction owner：直接从 spec/C、
固定 intermediate bounds、六处 compressed α 和模型参数构造 terminal seed，调用现有 full pipeline，
返回 lower 与所需 solver state，绕过每次原生 deque traversal、逐 node Python dispatch、A/bias list
装配和 root concretize glue。必须先捕获并冻结五次调用的输入/输出/effect，再做 replacement；不能通过
跳过状态更新伪造性能。

本轮不请求外审。只有新的高层 transaction replacement 完成三组以上 fresh correctness/performance 后，
再统一整理一次里程碑材料。

## 7. 验证

- root/BAB4 专项：`46 passed`；
- 全量回归：`2228 passed, 4 skipped`；
- 4 个 skip 均为既有环境边界：重复 TVM smoke、缺冻结 VNN-COMP checkout 两项、未配置 cuDNN root；
- Black：通过；
- mypy（7 个触及文件）：clean；
- Pylint：`10.00/10`；
- `git diff --check`：通过；
- 三组 fresh same-solver：退出码 0，环境全部 admitted，summary hash
  `273eb14dc6ae2ea0be21d1f99c2c77cbb572e502e771ea68a727e7174afdfd98`。
