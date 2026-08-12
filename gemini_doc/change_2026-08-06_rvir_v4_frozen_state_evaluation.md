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
