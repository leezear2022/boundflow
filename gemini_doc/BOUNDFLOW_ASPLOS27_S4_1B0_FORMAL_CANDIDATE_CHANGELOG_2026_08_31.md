---
status: formal-candidate-pending-external-audit
date: 2026-08-31
stage: s04
performance-claimed: false
---

# S4-1B0 formal correctness candidate 修改记录

## 结果

在已推送 source revision `4e2a26128a9a538ac64f222e8b82e92ea745d3b6` 上，从空正式路径生成
S4-1B0 correctness artifact：

- artifact：`artifacts/asplos27-s4-1b0-ternary/resnet2b-prop0-v1`；
- 5 positive + 1 cache + 5 fault，共11个独立进程；
- 5份positive sidecar均为`313,344 B`且SHA256全同：
  `a07aea90d2404b0e3c40f2af4aeaea169a1465b5feb24616c75cf882b5db5e6c`；
- selector计数：positive `8689`、negative `9137`、zero `606`、invalid `0`；
- selected output逐元素bitwise exact，旧binary rule会错分的zero为`606`；
- cache事件=`miss → hit`，compile/miss/hit/entry=`1/1/1/1`；
- 5类故障均在launch前以冻结reason拒绝；
- stdlib-only replay：PASS；
- 10类coherent outer-resigned tamper：`10/10 rejected`；
- targeted：`22 passed`；
- full suite：`2073 passed, 3 skipped`，耗时`715.10s`；
- Black、mypy、Pylint `10.00/10`通过。

## 证据身份

- manifest SHA256：`95a65429e4b59c0554c04f62b1b91f8538bc699bac809972aed173b009c43d76`；
- summary SHA256：`49f77c581bc9d96423da0e6e4e47da9714a6d8048c01160a5d2980c480c3244f`；
- summary hash：`d230415af6b0eaf81cd29c7a2c826ba1a443275104a56c6d4d38d5f0e096108f`；
- tamper report SHA256：`26f27868e04e7c218fdb6b988cb40c4b61b0b162490d07c3e37593ca9f82b4d5`；
- module receipt hash：`9ed4f12f787c5a6675728fad01526b2782ebef0d7045f7f4ba3a81e6776d57ac`。

TVM原样导出的`device_source.cu`以两个换行结束；为保留receipt绑定的原始字节而不改写冻结证据，
`.gitattributes`只对该artifact路径关闭`blank-at-eof` whitespace诊断，其余whitespace检查不变。

## 失败演练与修正

第一次临时演练中，coefficient最低有效位攻击没有改变符号，旧replay又缺少5份fresh sidecar全同回绑，
因此只拒绝`9/10`。正式数据生成前已在`4e2a261`中补上：

1. positive fresh-process binary determinism；
2. summary sidecar hash/byte count回绑raw；
3. coefficient攻击改为翻转IEEE sign bit以触发派生selector语义。

修正后从空临时目录完整重跑为`10/10`，随后才生成正式artifact。第一次演练目录已删除，未进入证据包。

## Claim边界

本轮只形成`FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1B0`。它不证明production evaluator
binding、optimizer trajectory、arena alias、timing、performance、same-solver、complete-query或10x；这些门禁
全部保持关闭。只有独立外审批准后才能写`VALIDATED-S4-1B0-TERNARY-ENDPOINT`并讨论是否开放S4-1B。
