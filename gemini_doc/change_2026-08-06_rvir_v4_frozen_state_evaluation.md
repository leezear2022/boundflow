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
