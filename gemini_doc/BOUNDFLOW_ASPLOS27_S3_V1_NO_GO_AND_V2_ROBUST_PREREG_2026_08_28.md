---
status: preregistered-s3-v2-robust-formal
updated: 2026-08-28T14:15:00+08:00
type: diagnosis-and-plan
topic: boundflow
slug: asplos27-s3-v1-no-go-v2-robust
stage: s03
performance-claimed: false
---

# S3 v1 NO-GO 诊断与 v2 稳健 formal 预注册

## 1. v1 必须保留的结论

source-exact=`2ae68b65b02e34a043a84bd08a11a14f10ce45cb` 的六 fresh v1 按冻结门禁得到：

- P/native geomean=`2.569574644743942x`；
- worst=`0.7595404807250521x`；
- P/旧 D2B geomean=`1.9500804966538938x`；
- verdict=`VALIDATED-NO-GO-S3-V1`。

不能删除第 5 个 worker、改 geomean 或降低门槛。v1 作为负结果永久保留。

## 2. raw 诊断

五个 worker 的 P/N 为 `2.996x—3.385x`，P 中位耗时 `30.50—36.03 ms`。`PND` worker 的 30 个 P
样本全部稳定在 `212.15—214.52 ms`，同时 D 从其他进程的 `56.74—62.34 ms` 变为 `531.68 ms`，N 从
`102.05—112.56 ms` 变为 `161.48 ms`。这不是单点 outlier，也不是只让 candidate 变慢的确定性 order
effect；相邻同样以 P 开头的 `PDN` 恢复为 `31.01/60.98/102.69 ms`。

因此 v1 同时证明两件事：当前 candidate 在健康进程中接近/超过 3x；单个 fresh-process-per-order 统计量对
一次持续性的机器状态扰动不稳健。此诊断不推翻 v1 NO-GO，也不形成性能 claim。

## 3. v2 冻结设计

v2 不重跑或替换 v1 的任一行，而是全新空目录、全新 source commit、全新 protocol：

- 六个顺序各运行 3 个 fresh subprocess，共 18 个 worker；
- 每 worker 仍为 5 warmup + 30 measured，内部规则、N/D/P 路径、语义容差与计时边界不变；
- 所有 18 个 worker 全部进入 raw，不允许删行、补跑单个失败项或按结果挑选；
- 对每个 order 的 3 个 pair speedup 取中位数，headline 为 6 个 order-median 的 geomean 与 worst；
- 同时披露 18-worker raw geomean、raw worst、每个 order 的三值和 dispersion；
- correctness、receipt、memory 对 18/18 全部执行，不使用中位数豁免；
- 2026-08-28 teardown smoke 发现连续约六个进程后 GPU 可进入持续慢功耗态；在新的 source-exact formal
  开始前追加冻结每个 fresh worker 之间 `15 s` 的 untimed cooldown。该等待不进入任何 latency 样本，且对
  N/D/P 三方一视同仁；protocol 必须记录该值；
- 3x gate 保持 `geomean>=3.00x / worst-order-median>=2.50x / P-D>=1.50x`；
- reduced/no-go 门槛不变；tamper、manifest、source blob 与 claim 边界不变。

选择每 order 三重复中位数不是删除 outlier：三条 raw 全保留且 estimator 在运行前冻结。若某种 order 的多数
进程仍退化，其中位数会失败；若 18-worker correctness 任一失败，整体直接 NO-GO。

## 4. 开放边界

只有 v2 通过全部门禁，才可关闭为 `VALIDATED-S3-3X-LOCAL-OPTIMIZER` 并开放 S4 same-solver implementation。
v1 与 v2 必须在 closure 中并列披露，禁止只报 v2。若 v2 仍不过，S3 关闭并进入 residual attribution，不能
继续增加重复次数或更换 estimator。

补充诊断依据：同一次六进程 smoke 的第六个 PDN 为 N/D/P=`158.56/531.10/212.19 ms`；固定等待 15 秒后
独立 PDN 恢复为 `101.63/57.20/30.90 ms`。这只用于冻结环境恢复间隔，不进入 headline。
