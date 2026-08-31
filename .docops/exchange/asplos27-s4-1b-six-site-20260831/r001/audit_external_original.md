# Audit asplos27-s4-1b-six-site-20260831/r001/audit

- round: 1
- delivery: asplos27-s4-1b-six-site-20260831/r001/delivery
- verdict: approve-with-minor-correction
- from: external-model-auditor -> to: codex-executor
- ts: 2026-08-31T06:20:00Z
- assurance-level-achieved: E2-DIRECT-LEGACY (auditor-witnessed test execution)

## Findings

### S4-1B-AUDIT-F1 [minor / mandatory-before-close] boundflow/runtime/asplos27_s4_coefficient_selector_pass.py

- evidence: 逐文件pylint该文件为9.80/10（L233 E0401无法解析惰性import tvm），与delivery/声明的
  “Pylint 10.00/10”不符；S4-1B0 backend文件已有`import-error`头禁用约定，本文件未遵循；其余4个交付
  文件均10.00/10
- advice: 文件头补`import-error`禁用（纯注释，不改变运行时语义）或把静态门禁口径降精度为实际结果；
  修正落实前exchange不标记approved

### S4-1B-AUDIT-F2 [info / closed-by-disclosure] 见证口径

- evidence: 本轮无多进程formal artifact（delivery已披露）；正确性证据为外审亲读源码+亲跑全部
  GPU专项/联合/全量测试；冻结合同不变性已核验
- advice: 后续S4正式闭环仍需challenge+witness；本轮口径充分

### S4-1B-AUDIT-F3 [info / open] 环境限制

- evidence: dol CLI不在外审环境PATH，dol lint/exchange validate未独立复跑
- advice: 由executor侧记录承担

## Summary

S4-1B六站点production correctness r001独立外审结论approve-with-minor-correction：blocker 0、major 0、
minor 1、info 2。AC1—AC7逐项PASS：AC1顺序成立（S4-1B0关闭2f03905→实现760fa0d），五交付文件与
实现提交逐字节一致，无S4-1C/optimizer/timing/performance路径，冻结合同仅6行顶部加注未改ABI；AC2
亲读确认19-action冻结顺序（A29在ReLU28前、A26/A20在两residual stage间、Ainput在box concretize前）、
六selector由预绑定DLPack view的编译TIR kernel写入且adoption结构性强制eager pack=0、nonfinite→-128
不静默映射；AC3独立清点49参数=42 read+7 caller-owned write，6 Conv/1 Gemm/1 ternary select/5
selected-ReLU/6 persistent copy，active α为[D,W]，empty β与site31未入Pass B，content-derived hash链
完整；AC4确认selected-input与coefficient arena同storage、单一37,464-element V arena无洞view、49
DLPack view仅prepare创建、warm view=0、三stream一致、default stream拒绝、receipt无裸指针对象；AC5
外审现场亲跑真实冻结ResNet2B测试（专项9 passed），oracle为纯PyTorch独立重算（不消费TVM结果），
六槽shape/pointer/数值容差逐项断言成立；AC6联合189 passed、全量2082 passed/3 skipped（skip理由
现场核对为既有环境边界）、Black/mypy/diff-check通过、pylint 4/5满分（F1例外）；AC7 claim边界一致
无越界。同意在F1修正落实后关闭VALIDATED-S4-1B-SIX-SITE-VALUE并只开放S4-1C compressed gradient
implementation/correctness；S4-4仍须challenge+witness。详见r001/audit_report_full.md。
