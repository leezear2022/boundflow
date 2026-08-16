# FSG4/B4-0 外部审计关闭记录

日期：2026-08-16  
最终状态：`EXTERNALLY-APPROVED-VALIDATED-B4-0-OPPORTUNITY`  
exchange：`fsg4-b4-0-kernel-attribution-20260816`，Round 1，`closed/approved`

## 1. 审计结论

外部模型不采信summary，从formal gzip raw用标准库独立复算AC1—AC7，全部PASS；无blocker/major，
2项minor与2项info均不阻塞。executor已执行`dol exchange close`，exchange validate与
`dol lint --soft`均PASS。

审计独立确认：

- source/code/B3 manifest/模型/property/三外部仓库与13文件manifest hash链全部一致；
- 270609 events、35367/35367 CUDA kernel closure，correlation/temporal=`33060/2307`，0丢失；
- 14 CROWN + 4 forward ordinal exact，CROWN/forward phase内temporal fallback=`0`；
- semantic discrete/sign exact，lower max diff=`4.76837158203125e-07`；
- CROWN14=`9196 kernels / 32618329 ns / 3291 mat ops / 57292800 B`，冻结share换算=
  `0.6771722591159042` B3 core；required region speedup=`3.989702826086512x`；
- terminal export与optimizer第10次evaluation具有完整重复call结构证据；
- root replay一致，executor 9/9 tamper拒绝；审计方自建第10类全重签allocation-delta攻击仍被拒绝；
- targeted/related/full=`15/54/1329 passed`，全FSG4 B3+B4并集=`96 passed`，静态门禁全过；
- 无speedup、B0 parity、memory saving或ASPLOS-ready claim漂移。

## 2. Findings转为B4-A硬门禁

1. CUDA kernel行shape为空是torch profiler边界。B4-A/B production shape不得从kernel行猜测，必须
   从同一correlation parent CPU operator行恢复，并绑定operator ordinal/name/shape/dtype/layout与
   lineage hash；缺失或多义时fail closed。
2. 后续exchange必须在request中列出related pytest的完整文件清单，不再只写聚合数字。
3. 67.72%是B3 span wall-share换算，不是CROWN14 kernel-only wall share；文档和artifact继续保留
   68.3% region kernel-sum占比与非CROWN工作边界。
4. CPU/CUDA时钟域不得用于跨phase ordinal排序；ordinal只由显式marker所有权确定。

## 3. 路由

只开放B4-A terminal lower/lA handoff：复用optimizer第10次、无update evaluation的terminal lower与
六层lower adjoints，消除terminal export重复CROWN call。B4-A必须自行完成typed handoff、数值/梯度
正确性、5 fresh pairs及B3/B4-A门禁；B4-B可设计但不得混入，B4-C/D与B5—B7继续关闭。

下一工程动作：先冻结B4-A预注册协议，不直接修改TIR。
