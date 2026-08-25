---
status: implemented-pre-formal
updated: 2026-08-26T16:30:00+08:00
type: changelog
topic: boundflow
slug: mr3-production-bridge-timing-formal-gates
stage: s01
---

# MR3 Production Bridge Timing Formal Gates 修改记录

## 1. 修改

- 新增`boundflow/runtime/mr3_production_bridge_timing.py`：冻结12-run顺序、worker/source/protocol、
  semantic、module、launch、device/stream、latency与absolute peak memory机械校验；
- 实现6 pair host speedup、geomean、固定seed 10,000次bootstrap 95% lower、worst pair、
  host/event方向一致性与allocated/reserved worst ratio；
- GO/NO-GO只由预注册阈值机械生成，只有全门禁GO才允许
  `performance_claimed=true`和`same_solver_complete_query_timing_open=true`；
- 新增raw-first formal runner：先replay MR3 correctness，拒绝已有/partial artifact，顺序启动12个
  独立进程，冻结raw后才派生summary/pairs/manifest；
- 新增tamper probe：16类worker+raw+outer manifest全重签非法变体必须全部拒绝；
- 新增synthetic unit tests，覆盖GO、NO-GO、source freeze与16类负路径。

## 2. 关键边界

- 任意正的、结构合法且重新签名的物理latency本身不能靠hash证明真假；tamper门禁针对可机械判定的
  source、结构、语义、计数、时钟、显存与派生不一致，不宣称密码学防恶意伪造计时；
- CUDA event只诊断方向，不参与headline或overlap adjustment；
- 本提交不含formal raw，不能形成性能结论。

## 3. 验证

- synthetic/negative：18项（GO/NO-GO、source freeze、16 tamper）预期全部通过；
- Black、mypy、pylint与`git diff --check`纳入提交前门禁；
- formal GPU结果留到clean implementation commit之后运行。
