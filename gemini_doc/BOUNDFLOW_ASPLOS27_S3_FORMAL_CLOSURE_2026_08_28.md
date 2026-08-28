---
status: internally-validated-pending-external-audit
date: 2026-08-28
type: formal-closure
topic: boundflow
slug: asplos27-s3-formal-closure
stage: s03
source-revision: 1766cbcbb95466f3c4d9afda448a5e1db9bfbe36
performance-claimed: false
same-solver-claimed: false
complete-query-claimed: false
tenx-claimed: false
---

# ASPLOS'27 S3 optimizer/runtime 正式收口

## 1. 结论

S3 v2 在冻结的 ResNet2B P-anchor、10 次 CROWN evaluation / 9 次 Adam mutation 本地 wrapper 上通过
`VALIDATED-S3-3X-LOCAL-OPTIMIZER-V2`：

- P/native 六个 order-median 的 geomean=`3.243894370020976x`；
- P/native worst order-median=`3.2246091003383275x`；
- P/旧 D2B geomean=`1.842216427387417x`；
- 18 个未筛选 raw pair 的 geomean/worst=`3.2478466674781026x/3.178493381578001x`；
- 18/18 逐步语义、结构 receipt、replay 与篡改门禁通过。

这只开放 S4 same-solver **实现与正确性接入**。它不是 same-solver、complete-query、跨模型、总体 10x 或
ASPLOS-ready 性能结论；artifact 中这些 claim flag 均保持 `false`。

## 2. 为什么保留 v1 NO-GO

source=`2ae68b65b02e34a043a84bd08a11a14f10ce45cb` 的 v1 六 fresh 仍是有效负结果：

- P/native geomean=`2.569574644743942x`；
- worst=`0.7595404807250521x`；
- P/旧 D2B geomean=`1.9500804966538938x`；
- verdict=`VALIDATED-NO-GO-S3-V1`。

v1 的第五个 worker 在整个进程内同时让 N/D/P 持续变慢，不是 candidate 单点异常。v2 没有删除、替换或
续跑该行，而是在新 source、新目录中执行预先冻结的 3 replicate/order、18 fresh protocol，并在每个 worker
之间加入计时外 15 秒 GPU 状态恢复间隔。v1 artifact 与三个后续崩溃失败尝试全部保留。

## 3. v2 正式协议与结果

### 3.1 协议

- 六个执行顺序：`NDP/NPD/DNP/DPN/PND/PDN`；
- 每个顺序 3 个 fresh subprocess，共 18 worker；
- 每 worker 每路径 5 个完整 10/9 warmup + 30 个完整 10/9 sample；
- 所有 raw 行进入证据，不删除 outlier、不补跑单项；
- 每个 order 的 3 个 pair speedup 先取中位数，再对六个 order 求 geomean/worst；
- 门槛保持 P/N geomean `>=3.00x`、worst `>=2.50x`、P/D geomean `>=1.50x`。

### 3.2 六个 order-median

| order | P/native | P/旧 D2B |
|---|---:|---:|
| NDP | 3.3058681213x | 1.8524836100x |
| NPD | 3.2317970188x | 1.8356199898x |
| DNP | 3.2416591918x | 1.8220192035x |
| DPN | 3.2246091003x | 1.8493506153x |
| PND | 3.2270188353x | 1.8560982439x |
| PDN | 3.2331413635x | 1.8379495461x |

candidate 的 18 个中位 latency 位于 `30.0892705–31.8899305 ms`。全部三重复原始 ratio 与 latency 保存在
v2 `summary.json` 和 `raw/workers.jsonl`，表格不代替 raw。

## 4. 数值与状态等价

18 个 worker 都从同一冻结 pre-state 独立建立 N/D/P owner。replay 对 10 个 ordinal 的 lower、compressed
dα、α before/after、Adam step/m/v、scheduler 与 terminal state 逐项重算。全体最大绝对误差：

| 字段 | max abs diff | 门槛 |
|---|---:|---:|
| lower | `7.867813110351562e-06` | `2e-4` |
| compressed dα | `8.288770914077759e-08` | `2e-5` |
| α before/after | `4.917383193969727e-07` | `2e-5` |
| Adam exp_avg | `4.190951585769653e-08` | `2e-5` |
| Adam exp_avg_sq | `1.057287590811029e-11` | `2e-5` |

lower 与 gradient sign 全部 exact。P 的 evaluation/mutation=`10/9`；host 仍拥有 Adam、clamp、scheduler 与
每一步 policy cut，未把一次冻结轨迹冒充通用 optimizer IR 或无条件 device loop。

## 5. 生命周期修复与预注册偏离

成功 formal 前保留了三个完整失败尝试：

1. attempt A：第 13 个 worker `SIGABRT`，促使 harness 先写失败日志再 fail closed；
2. attempt B：成功 worker 退出时出现 TVM allocator double-free；
3. attempt C：成功 worker 退出时出现 glibc heap corruption。

根因有两层：selected-value CUDA Graph 捕获了 Relax/cuDNN 临时 workspace 指针；同时 TVM cuDNN
`ConvEntry` 把长生命周期 workspace 放进 TLS 临时 pool，进程退出时两个 TLS owner 的析构顺序未定义。
正式 source 在**运行 v2 前**完成并冻结以下修复：

- selected-value Relax/cuDNN 不再进入 CUDA Graph，每 ordinal 安全执行一次 VM；
- 结果通过第 5 个 inplace-copy TIR 写入 persistent output，warm DLPack view=`0`；
- 外层只含 persistent tensor/TIR 的 forward graph 保留；
- vendored TVM `ConvEntry` 改用匹配的 `AllocDataSpace/FreeDataSpace` 持久 workspace owner；
- receipt 强制 `selected graph=0 / VM=10 / output-copy=10 / warm DLPack=0`。

这取代了原 S3 预注册中“selected graph replay=10”的不安全结构预期，但没有降低数值或性能门槛。变更在
source-exact commit及 change log 中先于最终 formal；v2 replay 也强制拒绝旧 receipt。历史 S2 的数值结果
仍保留，但其“selected-value CUDA Graph 安全可复用”机制表述必须由本节修正，不能继续作为 production claim。

## 6. 内存口径

v2 whole-wrapper 的 warm peak dynamic allocated/reserved 最大值为 `13,824/0 B`，不能写成 `0/0`。
`13,824 B` 在18/18完全一致，来自每次 wrapper 新建的 host-owned Adam 首步梯度与 moment 状态；它属于本轮
刻意保留的 optimizer policy，不是 compiled CROWN region 恢复了动态 output 或 DLPack allocation。prepared
selected output、TIR buffers 与 cuDNN workspace 都在 prepare/lifetime owner 内建立。

因此本轮只得出“无新增 CUDA reserved block、compiled region warm dynamic output/DLPack owner为零”，不主张
whole-wrapper dynamic allocated=`0`，也不形成系统 memory claim。

## 7. Replay、tamper 与测试

- v2 replay：PASS，summary hash=
  `494feff6457da88e45cf9a4906d42fac2254d6d4323d8d90732503ba6860fb6d`；
- 10 类 fully outer-resigned tamper：`10/10 rejected`，覆盖 latency、step tensor、optimizer moment、
  replicate identity、execution counter、estimator、gate、claim flag、code revision 与 summary status；
- v1 replay继续通过，并保持NO-GO；
- targeted环境+S2+S3：`19 passed`；
- full regression：`1884 passed, 3 skipped, 6 warnings`；三个skip分别为TVM已存在时跳过重复无TVM
  编译，以及两项缺冻结VNN-COMP checkout的既有环境边界；
- Black：12个相关Python文件无需改动；mypy：12 files clean；pylint：`10.00/10`；
- `git diff --check`：PASS；DocOps lint见最终validation记录。

## 8. 证据入口

- v2 artifact：`artifacts/asplos27-s3-optimizer/resnet2b-p-anchor-v2`；
- v1 NO-GO：`artifacts/asplos27-s3-optimizer/resnet2b-p-anchor-v1`；
- 三个失败尝试：同目录下`v2-failed-attempt-a/b/c`；
- protocol/replay：`scripts/run_asplos27_s3_optimizer_artifact_v2.py`；
- tamper：`scripts/probe_asplos27_s3_optimizer_v2_tamper.py`；
- runtime：`boundflow/runtime/asplos27_s3_optimizer_pipeline.py`；
- selected-value lowering/runtime：`boundflow/backends/tvm/asplos27_s2_selected_value.py`、
  `boundflow/runtime/asplos27_s2_crown_pipeline.py`；
- TVM workspace owner：submodule commit=`9802f45b802225f2ea46499eec4ab7b16f64a73f`。

## 9. 唯一后继

只开放 S4 implementation/correctness：由 RVIR adapter 在同一个 αβ-CROWN host solver 内，用
`PreparedBoundProgram`替换对应 exact-call bound executor，保持 branch、termination、state trajectory、seed、
device和dtype不变。必须先测真实same-solver share与integration overhead，再决定性能formal；不得把本轮
`3.2439x`直接外推为query或总体收益。
