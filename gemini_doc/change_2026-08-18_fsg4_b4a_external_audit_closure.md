# FSG4/B4-A 外部审计关闭记录

日期：2026-08-18

最终状态：`EXTERNALLY-APPROVED-VALIDATED-NO-GO-B4-A-PERFORMANCE`

exchange：`fsg4-b4a-formal-timing-20260818`，Round 1，`closed/approved`

## 1. 审计结论

外部模型未采信summary，独立复核hash链、24个worker raw、correctness/activation/environment/
profile、ratio、root replay、14类outer-resigned tamper、回归与静态门禁。AC1—AC7全部
PASS，0 blocker / 0 major / 1 minor / 1 info。executor已接受findings并执行
`dol exchange close`。

外审独立确认：

- 24/24 environment admitted，thermal/power interval delta严格耦合；
- 6/6 correctness pair和每对19个tensor通过，全局max diff=`4.4107437e-06`；
- core wall geomean=`1.0189949992x < 1.03x`；
- query worst=`0.9969470224x >= 0.98x`；
- replay PASS，14/14 outer-resigned tamper全部语义拒绝；
- related=`73 passed`，full=`1356 passed, 3 skipped`。

因此B4-A的唯一合法分类是`VALIDATED-NO-GO-B4-A-PERFORMANCE`。B4-A只保留
correctness/mechanism evidence，约1.9% core改善不得计入B4 cumulative performance
baseline，不支持memory、B0 parity或ASPLOS-ready claim。

## 2. Findings处置

1. frozen v5 artifact中hash-bound raw stdout含尾随空格，一个预注册文档含EOF空行；
   不重写immutable raw。独立的11个Python文件scoped `git diff --check`为PASS。
2. Mypy的正确口径是5个产品/runner脚本以`--explicit-package-bases`通过；额外将
   6个测试文件纳入会产生24个测试侧typing diagnostics。后续validation evidence必须列出
   精确文件清单与参数。

两项finding均不影响artifact、replay或NO-GO分类。

## 3. 路由

仅开放单独预注册B4-B differentiable lower-only CUDA/TIR。B4-B必须自己冻结production
shape、可微状态所有权、forward/backward parity、activation receipt和局部/全局claim边界；不得
放宽PR-12 plain-CROWN capability，不得把B4-A的1.9%改善带入B4-B基线。

完整外审证据：
`.docops/exchange/fsg4-b4a-formal-timing-20260818/r001/audit_report_full.md`。
