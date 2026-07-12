# PR-10 Materialization Profile：第一版证据摘要

> 运行：`pr10-profile-full-clean-v3-20260712`  
> 代码提交：`8f2c998`，`git_dirty=false`  
> 硬件：RTX 4060 Laptop GPU，PyTorch 2.12.1+cu132  
> 结果：180/180 `ok`，无 fail/OOM/unsupported

> 注意：该运行早于 structured 主路径切换，全部是 dense reference baseline。新的双模式结果
> 由后续 comparison artifact 单独记录，本文件中的数字不能当作 structured 结果。

## 1. 矩阵与口径

- workload：MLP chain、CNN chain、residual block、add+concat DAG、三 BasicBlock mini-ResNet；
- method：CROWN、α-CROWN、固定 split αβ-CROWN；
- spec batch：1、9、32、128；
- domain batch：1、8、32；当前是 `synthetic_fixed_domain_batch`，不是 BaB 树抽样；
- α/β optimization steps：1；warmup 1、timing repeats 3；
- latency 来自 trace-off；peak 来自清空 CUDA cache 后的独立 trace-off 单次执行；
- trace-on 只用于 event、logical bytes、operator tree 与逻辑 lifetime。

原始证据位于（按仓库规则不提交 Git）：

```text
artifacts/phase7a-pr10/pr10-profile-full-clean-v3-20260712/profile/raw.jsonl
artifacts/phase7a-pr10/pr10-profile-full-clean-v3-20260712/profile/normalized.csv
artifacts/phase7a-pr10/pr10-profile-full-clean-v3-20260712/profile/manifest.json
```

## 2. CROWN 在最大扫描点（spec=128、domain=32）

| workload | events | logical bytes | max event | peak allocated | median ms |
|---|---:|---:|---:|---:|---:|
| MLP chain | 4 | 8,388,608 | 2,097,152 | 50,706,432 | 6.34 |
| CNN chain | 4 | 100,663,296 | 33,554,432 | 628,050,944 | 15.35 |
| residual block | 6 | 201,326,592 | 33,554,432 | 1,086,989,312 | 53.49 |
| add+concat DAG | 6 | 100,663,296 | 16,777,216 | 597,048,832 | 31.60 |
| mini-ResNet | 14 | 469,762,048 | 33,554,432 | 2,311,318,016 | 65.03 |

mini-ResNet CROWN 的 event 数随 batch 不变，但 logical bytes 从 spec=1/domain=1 的 114,688
增长到 spec=128/domain=32 的 469,762,048，正好是 `128 × 32 = 4096` 倍。这证明当前
persistent ReLU coefficient fallback 直接随 query axes 放大。

## 3. αβ 代表点

mini-ResNet αβ、spec=128、domain=32：

```text
events                         28
logical materialized bytes     939,524,096
max event                       33,554,432
peak CUDA allocated              3,445,742,080
peak CUDA reserved               3,529,506,816
alpha state bytes                1,835,008
beta state bytes                 1,835,008
intermediate bound bytes         9,439,744
weight bytes                        96,872
median trace-off latency             157.14 ms
```

event 数是 CROWN 的两倍，因为配置中的一次 αβ optimization 产生两次 backward evaluation。
该数字不能被解释为算法无关的 method 对比。

## 4. 当前可支持与不可支持的 claim

可以支持：

- `C1-E1a`：ReLU persistent dense fallback 在 CNN/DAG/mini-ResNet 上可观测，且 logical bytes
  随 spec×domain 严格放大；
- `C1-E1b`：在最大 mini-ResNet 点，累计 logical coefficient materialization 达 0.94 GB，
  trace-off peak allocated 达 3.45 GB，存在明确优化空间；
- trace schema 能区分 coefficient、α、β、intermediate、weight 与 allocator peak。

尚不能支持：

- “ReLU 是完整 verifier 的主导内存瓶颈”：logical cumulative bytes 与 allocator peak 不是同一口径；
- “Planner 必然优于固定策略”：尚未比较不同 materialization plan；
- “domain batch 结果代表真实 BaB”：当前域是合成固定域批；
- 任何 structured ReLU 的性能/显存收益：operator 尚未实现。

## 5. 决策

Opportunity Gate 对继续 PR-10 为 **GO**：非 toy mini-ResNet 已出现随 query axes 放大的
persistent dense state，值得建立 dense/gradient oracle 并实现精确 SignSplit operator。
PR-11 Planner 仍未获得 Go；必须等 structured/eager 策略对照出现至少两个不同 regime。
