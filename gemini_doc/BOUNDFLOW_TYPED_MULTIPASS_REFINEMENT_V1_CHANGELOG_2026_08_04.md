# BoundFlow Typed Multi-Pass Refinement v1 修改记录（NRIR-26）

日期：2026-08-04
分支：`feat/typed-multipass-refinement-v1`
状态：`VALIDATED-NO-GO`

## 当前记录

- PR #36 已合并为 `main@78ffa6b`，从该基线建立 NRIR-26 分支；
- 冻结 same-total-cap 的 single-pass dynamic8_24 与 split-two-pass dynamic8_24 对照；
- 冻结 objective influence、updated-width re-score、prior-target exclusion 与 no-unseen-target stop；
- 要求 multi-pass control 明确进入 Plan/Task/Schedule 与 pass trace，不接受仅在 Python runtime 中
  把同一 targets 多跑一次；
- 固定 ResNet clauses `0/2/4`、31 nodes/depth 4、CPU 六分片 fresh replay，禁止性能或完整验证宣称。

## 待完成

- typed multi-pass policy/decision 已实现：总 cap 等分、updated-width 排序、prior-target ledger、
  no-unseen stop、逐 pass stable hash 与自包含 lineage validation；
- multi-pass Plan lower 为两组显式
  `enumerate→select/decide→backward→intersect→propagate` Task/Schedule；legacy lowering/hash 分支
  保持不变；runtime 对空 target stop 做 sound passthrough并记录 decision/action/pass trace；
- optimized queue 已把 multi-pass policy/decisions 绑定到逐 node refinement record、execution 与顶层
  queue trace；dynamic assigned 8/16/24 cap 在每 node 精确拆为 4+4/8+8/12+12；
- 新增 direct IR 的 disjoint reselection、stop/passthrough、decision tamper tests，以及 queue 的逐 node
  lowering、dynamic partition、admission/tamper tests；新路径 `4 passed`，legacy refinement/queue
  `31 passed`；targeted Mypy clean；
- 六个 fresh-process shards 已生成并通过静态校验；artifact evidence hash=
  `38992cace70214ffcbd670f03dcfca182e0925bee31eb4df885dab4dab03494d`；
- clauses `0/2/4` 的 single 与 split-two-pass worst terminal lower 均分别为
  `-0.2819737196/-0.4016119838/-0.4596676826`，三条 delta 全为 `0.0`；logical-domain
  overlap/union 全为 `31/31`；
- 两 mode 每 clause 的 planned total cap 均为 `496`，actual selected target count 均为 `2976`，
  split mode stopped pass=`0`；说明第二 pass 选满且 ledger/预算生效，但没有改变关键 worst domain；
- 因未满足“至少一条严格改善”的预注册门禁，方法结论为 `VALIDATED-NO-GO`；typed IR/control
  mechanism 保留，不升级 tightness/property claim；
- artifact tests 固定 digest、三 clause 零 delta、逐 pass cap/ledger、program/decision/claim tamper
  与 checkpoint fail-closed；fresh-process semantic replay 6/6 通过，最终输出
  `{"evidence_hash":"38992cac...494d","status":"ok"}`；
- focused `50 passed`；全量 `787 passed, 37 skipped`（skip 均为 CUDA/环境门禁）；Black、targeted
  Mypy、Pylint `10.00/10` 与 `git diff --check` 通过。

## 下一路线

- 停止继续测试“冻结 node-initial objective influence + 仅更新 width + 同总 target 集拆 pass”的
  变体；证据表明它没有改变 worst-domain proof deficit；
- 若继续 refinement，下一门禁必须改变信息内容而非执行顺序：pass-local objective influence
  recomputation 或与 branch lookahead 联合，并先做小范围可区分性探针；否则转向更高杠杆的 split/
  cut semantics。不得从本 NO-GO 推导 performance 或 ASPLOS-ready。
- 更新权威文档、全量/static/DocOps 验证并发布。
