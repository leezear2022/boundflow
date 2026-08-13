# RVIR-v4 V4-2C Pre-State Native Initializer 修改记录

日期：2026-08-13

## 结论

V4-2C第一切片已实现共享production→native pre-state mapper，状态为
`IMPLEMENTED-MAPPER-READY / FORMAL-ARTIFACT-PENDING`。它把V4-1原先函数内部的映射提升为独立typed
合同，并在V4-1 evaluator中复用同一实现，不再维护两套α/β/split布局逻辑。

本切片只证明pre-state可逆初始化，不执行10-step optimizer mutation，不做copy-out，不关闭V4-2或
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

## 验证与边界

- focused initializer + V4-1 reuse：`11 passed`；
- mypy三文件clean；Pylint三文件=`10.00/10`；
- full suite=`1162 passed, 3 skipped`；3项skip不含CUDA；
- 尚未生成V4-2C独立formal artifact，所以本提交不关闭V4-2C；
- `optimizer_replacement_admitted=false`、`b2_same_solver_timing_admitted=false`、
  `performance_claimed=false`。

## 下一步

从本轮clean implementation commit生成独立V4-2C artifact，replay原始V4-2B source artifact，冻结上述
identity/mapping/12 receipts，并执行同步重签名的topology、index、history、intermediate、upper-plane、
beta-location篡改探针。全部通过后才关闭V4-2C并进入V4-2D逐step mutation parity。
