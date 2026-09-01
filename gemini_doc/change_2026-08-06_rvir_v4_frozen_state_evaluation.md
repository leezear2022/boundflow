# RVIR-v4 Frozen-State Evaluation 修改记录

日期：2026-08-06

## 修改

- 新增native frozen-state mapper/evaluator，把production start-node alpha、sparse beta、split history、
  refined intermediate bounds映射到BoundFlow五层IR；
- 映射通过显式六层topology contract连接provider节点名和BoundFlow primal value；
- alpha按v2 coordinate indices恢复dense slope，beta按SparseBeta location恢复，split按domain history恢复；
- 新增真实ResNet2B artifact-backed回归测试，不调用αβ-CROWN provider。

## 验证

- production lower：6个child；
- native lower max abs diff=`2.0265579223632812e-06`；
- sign agreement=`6/6`；
- focused test=`1 passed`，mypy clean；
- `performance_claimed=false`。

## 边界

这证明同一真实core的冻结post α/β/split state可由BoundFlow native evaluator复算lower；不证明
BoundFlow能执行10-step optimizer mutation，也不产生性能claim。下一动作是生成V4-1正式artifact、
重放与state/topology tamper门禁；通过后才准入V4-2。

## Artifact Runner

新增正式generate/replay runner，固定v2 capture/source manifest/model digest与六层topology；replay会
重新执行native IR，而不是只校验JSON。topology JSON必须与代码内冻结映射一致；source capture必须与
代码内固定digest一致。runner提交后再生成正式工件，避免manifest绑定dirty source。

## Full-suite Replay Numeric Semantics 修正

- 首次全量回归在`1089 passed, 39 skipped, 1 failed`结束；唯一失败是正式replay把重新执行得到的
  `native_lower`浮点投影与冻结`execution.json`做整份字典exact比较；同一artifact的单独replay、
  focused replay和10个不同`PYTHONHASHSEED`均通过；
- replay合同修正为：shape、IR/state hash、计数、布尔门禁、production lower与字段集合保持exact；
  `native_lower`及其派生max-diff必须finite并按V4-1预注册`atol=rtol=2e-4`比较；
- artifact自身的summary canonical hash及manifest绑定仍exact，容差不允许篡改结构、状态或来源；
- 新增容差内数值漂移准入和`1e-2`容差外漂移拒绝测试。修正提交后必须重新生成绑定clean source的
  正式artifact并重跑全量回归，完成前V4-1仍不关闭。

## V4-1 正式关闭（2026-08-13）

状态：`VALIDATED-REDUCED-FROZEN-STATE-EVALUATION`。

- source commit=`c74a2049d3d2484aade7fd5b3dd805df53823d78`；
- artifact=`artifacts/rvir-v4-frozen-state/resnet2b-core-v1`；
- manifest/summary hash=`ba6ee2fc32109adc38326d58f7253a0cdeba2dd988ccb957f7a626d6544adf95`/
  `3541318b226ffd28cad0862e1b43055cc701d0973144cb58f4e17122a49f60e9`；
- frozen state hash=`8f8cd55d995ffcc1e14da353d944e8c2b73867e683a5d0af0fce6997a1b793fe`，
  topology hash=`9be361625e492b1401a402fd19ad5d80ac06a977c74f137c7563e96de06bca35`；
- 6个真实child：lower max abs diff=`2.0265579223632812e-06`、sign=`6/6`、五层IR相关
  hashes=`10`、replacement/original/fallback dispatch=`1/0/0`；
- 原样generate/replay通过；攻击方同步重签manifest后，topology semantic tamper与capture state
  tamper仍fail closed；数值漂移`1e-6`按预注册容差准入，`1e-2`被拒绝；
- focused=`21 passed`；全量回归=`1092 passed, 39 skipped`；mypy clean；Pylint=`10.00/10`；
- `performance_claimed=false`。V4-1只证明post-optimized frozen state的独立native复算；不证明
  optimizer mutation replacement。下一阶段只准入V4-2的10-step mutation预注册与实现，B2仍关闭。
