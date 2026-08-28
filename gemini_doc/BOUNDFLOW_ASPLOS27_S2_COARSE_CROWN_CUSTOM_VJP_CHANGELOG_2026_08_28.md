---
status: implementation-started
date: 2026-08-28
type: changelog
topic: boundflow
slug: asplos27-s2-coarse-crown-custom-vjp
external-audit: deferred-by-user
performance-claimed: false
---

# ASPLOS’27 S2 coarse CROWN + custom VJP changelog

## 2026-08-28 开工

- 用户要求继续实现，下一轮再统一外审；
- S1 external-audit exchange保留为已交付历史边界，不把后续S2写入该round；
- 只读单evaluation归因确认native约`9.002 ms`、旧D2B约`6.483 ms`；
- 旧D2B内部effective-value约`3.693 ms`为最大瓶颈，forward约`1.745 ms`、coefficient/sign约
  `0.869 ms`；
- S2第一刀改为standard Relax + TVM cuDNN重建selected-value Conv chain，不包裹旧serial
  `effective_pre23`；
- 本文后续连续记录实现、失败、correctness、formal与closure；`performance_claimed=false`。
