# RVIR-v4 V4-2C Pre-State Native Initializer 修改记录

日期：2026-08-13

## 结论

V4-2C已以`VALIDATED-PRE-STATE-INITIALIZER`关闭。共享production→native pre-state mapper把V4-1
原先函数内部的映射提升为独立typed合同，并在V4-1 evaluator中复用同一实现；正式artifact又从
V4-2B frozen source与ONNX独立重建真实native scope/state并通过双层篡改门禁。

本阶段只证明pre-state可逆初始化，不执行10-step optimizer mutation，不做copy-out，不关闭V4-2或
B2，也不产生性能claim。

## 实现

- 新增`rvir_v4_pre_state_initializer.py`：
  - 精确绑定6组provider activation/preactivation/start-node与native preactivation topology；
  - 按`alpha_indices`把lower polarity/start-spec `[0,0]`的compressed α散射为6组dense domain slope；
  - α的未消费upper polarity plane不丢弃，作为显式copy-through进入full round-trip receipt；
  - 把6组SparseBeta value按location散射为dense β，并从36项history恢复dense int8 split；
  - 绑定6组external intermediate lower/upper，分别冻结snapshot/topology/history/intermediate identity；
  - 12个mutable path分别生成mapped/full source→dense→compressed round-trip receipt；
  - mapper可在独立native scope上构造`NativeAlphaBetaOptimizationState`，不持有provider对象。
- V4-1 frozen-state evaluator改为调用共享initializer；历史V4-1数值和artifact replay保持通过。

## 正式pre-snapshot映射结果

输入来自V4-2B正式GPU artifact step 0绑定的core pre-snapshot：

- snapshot hash=`2a775b66559c20ddfc0bec97ec026898ba5eccfc984e02b217fcb7472d03a256`；
- topology hash=`9be361625e492b1401a402fd19ad5d80ac06a977c74f137c7563e96de06bca35`，
  与V4-1冻结拓扑一致；
- history hash=`8921a052baa3a1444c468851f9a8be6429b23830982a61ee285b2cb2b115a08a`；
- intermediate bounds hash=
  `f82523fb83031f5d0699dc5ff15078a7b6be1c0ca03511f2d53093721288cf06`；
- mapping hash=`cfcebf92fc58c269899d98cd65cc9454d7caa6051e2c9da46d415eda1fecf8df`；
- 6 α + 6 beta-value receipts全部mapped/full hash exact、max diff=0、sign exact；
- 6组α receipt均含非零copy-through元素，证明upper plane被保留；split非零总数=6。

## Fail-Closed 门禁

- wrong start-node导致mutable-path ownership不完整并拒绝；
- 重复native topology key拒绝；
- upper α plane虽不进入lower-only dense α，仍改变snapshot identity；full round trip保留它，而冻结
  expected identity拒绝漂移；
- snapshot与optimizer trace step 0的12项mutable path/content digest必须一一相同；
- topology、history coverage、intermediate/layout角色、α index ordinal/坐标唯一性、beta location唯一性/
  范围、split coefficient均显式校验。

## Mapper 实现提交时的验证与边界

- focused initializer + V4-1 reuse：`11 passed`；
- mypy三文件clean；Pylint三文件=`10.00/10`；
- full suite=`1162 passed, 3 skipped`；3项skip不含CUDA；
- 当时尚未生成V4-2C独立formal artifact，所以mapper实现提交本身不关闭V4-2C；
- `optimizer_replacement_admitted=false`、`b2_same_solver_timing_admitted=false`、
  `performance_claimed=false`。

## Mapper 提交后的动作（现已完成）

从本轮clean implementation commit生成独立V4-2C artifact，replay原始V4-2B source artifact，冻结上述
identity/mapping/12 receipts，并执行同步重签名的topology、index、history、intermediate、upper-plane、
beta-location篡改探针；这些动作已由下述Formal Closure完成。

## Formal Runner 准备（同日）

新增`run_rvir_v4_pre_state_artifact.py`及capture-ready测试。runner从V4-2B正式artifact重新校验并复制
source capture/manifest，导入冻结ResNet2B ONNX，在真实BoundFlow module上构造native scope，将mapper
结果恢复为`NativeAlphaBetaOptimizationState`，并冻结topology、mapping、native state、summary、源码
revision及完整文件inventory。replay不信任序列化结果，而是从source capture与ONNX重新执行映射并逐项
比较。当前focused验证=`7 passed`，mypy四文件clean，Pylint runner/test=`10.00/10`。

该runner需要先进入clean commit，随后才能生成带source revision的正式artifact；因此本段仍不升级
V4-2C状态，也不声明optimizer replacement或性能结果。

## Tamper Probe 准备（同日）

新增六类同步重哈希/重签名攻击：topology、α sparse index、history score、intermediate bound、未消费
upper-α plane、beta location+history。对source capture类攻击，probe会重算tensor digest、snapshot hash、
V4-2B source manifest及V4-2C outer manifest；同时绕开序列化结果直接调用semantic builder，要求冻结
identity或snapshot/step-zero cross-binding拒绝。临时报告实测6/6在外层provenance和内层semantic两层均
fail closed；mypy clean，Pylint=`10.00/10`。probe先进入clean commit，再生成其源码digest绑定的正式报告。

## Formal Closure（同日）

- clean runner commit=`96c45a6`；artifact路径=
  `artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/`；
- original semantic replay exit 0，mapping/native-state/summary hash=`cfcebf92...f8df`/
  `e3587dd9...bff0`/`6702a39d...899c`；artifact manifest SHA256=`daee2fa0...0218`；
- tamper report路径=`artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1-tamper-report.json`，6/6攻击
  同时通过outer provenance rejection和direct semantic rejection；报告SHA256=`894c30c4...d858`，
  report hash=`cfe3f9cd...0033`；
- focused formal=`8 passed`；full=`1164 passed, 3 skipped`，3项skip均为既有TVM/VNN checkout边界；
  mypy七文件clean，Pylint七文件=`10.00/10`。

V4-2C由此关闭。下一步只进入V4-2D：以同一native pre-state执行并逐step对齐10次evaluation、9次更新、
lower与α/β状态；不得提前执行copy-out、B2计时或性能宣称。
