---
status: implemented-b4b1-reference-artifact-runner-pending-formal
updated: 2026-08-18T15:27:00+08:00
type: change
topic: boundflow
stage: s01
---

# FSG4/B4-B1 five-fresh reference artifact runner 候选

## 改动

- 新增root runner，从hash-bound B4-B1a 5 fresh raw逐条重编译typed IR与instance；
- 每条raw独立执行pure-PyTorch forward/local VJP并生成hash-bound receipt；
- protocol绑定source artifact manifest/protocol/summary/run-file hashes、代码revision、α/β语义与
  tolerance；root replay重新读取raw和执行reference，不信任已存records/summary；
- 新增5 fresh静态IR稳定性与候选summary门禁。

## 候选结果

10 captures、60 tensor metrics、196,380 elements全部allclose且sign exact，最大误差=
`1.9073486328125e-06`；S β gradient=5/5、P incoming-A gradient=5/5；S/P静态IR hash各自
跨5 fresh唯一。仍标记`coordinated_rewrite_integrity=pending-separate-probe`。

## 下一步

提交runner后从clean source生成formal artifact，再实现协调动态bias/adjoint全重签篡改probe。
B4-B2/TIR/performance继续关闭。
