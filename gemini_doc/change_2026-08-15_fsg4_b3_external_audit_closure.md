# FSG4/B3 正式计时外部审计关闭记录

日期：2026-08-15

状态：`EXTERNALLY-APPROVED-VALIDATED-REDUCED-B3`

## 结论

DocOps exchange `fsg4-b3-formal-timing-20260814` Round 2 已由独立外部模型完成实质性审计并
`approve`，executor随后执行`exchange close`，最终状态=`closed/approved`。审计方未采信summary，
使用纯标准库脚本从36个raw worker独立重算44项检查，AC1—AC7全部PASS，无blocker/major/minor。

因此FSG4/B3正式关闭为`VALIDATED-REDUCED-B3`：只允许以B3作为累计基线启动B4 operator/cross-stage
CUDA/TIR fusion candidate；B5 JIT/CUDA Graph、B6 runtime streams、B7 arena/memory以及最终
`1.20x queue / 1.15x complete-query`门槛继续关闭。

## 审计证据

- exchange：`.docops/exchange/fsg4-b3-formal-timing-20260814/`；
- substantive audit：`r002/audit.md`与`r002/audit.json`；
- full report：`r002/audit_report_full.md`；
- closure：`closure.md`与`closure.json`；
- audit md/json/full-report SHA256：
  `5ec37ecb41f49200da0eefa7ebbd96f9481a0ee553504f0fb4114755b7596fb7`/
  `1365b6b3cfd25d64f74368aae5e9949e664fe1bfc27fee765a0c59b12bb8d2dd`/
  `a5b6c89aac45f1674dc4ccf4379810cd0c0b2b44951be67d7be8823757e14d2a`；
- closure md/json SHA256：
  `3bab7604bd08328b15b1bddd6c6958f6368c57ca1b9bd7aa2b169c6b495f5c8e`/
  `1c8d9220ef894037cd26649d607bfc1d81045ad17f009a749c8660e028058ee4`。

## 独立复核摘要

- source=`36e9069`、19个code revision、five-fresh manifest、模型/property digest及三个外部checkout
  revision全部独立核对一致；artifact零本机路径泄漏；
- 六全排列36 worker顺序逐元素一致，subprocess独立性与raw-first/resume拒绝门禁成立；
- 30+6 direct semantic pair、B0/B2/B3 execution ownership、12/12 B3 activation receipts及profile
  counters全部独立通过；
- 36/36 environment admitted，最大closure=`0.002510499028552414`，最大扰动=`1.043622<=1.05`；
- 独立重算B2/B3 core/query=`1.0716174805930418x/1.0066228954759742x`、B0/B3 query=
  `0.9100012637918488x`、最差core pair=`1.0635877032562384x`，分类恰为
  `VALIDATED-REDUCED-B3`；
- replay、10/10 outer-resigned tamper、frozen 6、targeted 114、full 1314+3 skipped全部复现；
- `performance_claimed=false`在完整链路保持，无memory或BoundFlow-over-auto_LiRPA claim。

## Findings 与处理

仅两条info，无需代码修复：

1. 正式solver使用αβ-CROWN venv的Python 3.11/torch 2.11，而开发Conda为Python 3.12/torch 2.12；
   36/36一致且进入raw，换环境会由protocol identity自然fail closed。
2. 上游solver stdout含行尾空格；它们是manifest-bound原始证据，继续按冻结豁免保留，不得改写。

## 下一步

只允许创建B4 cumulative fusion预注册和B4-0 attribution/qualification。B4必须以同一solver中的B3为
直接对照，同时持续报告B0累计结果；不得把B3相对B2的`1.071617x`当成B4收益，也不得提前启用B5—B7。
