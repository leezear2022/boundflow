# Audit asplos27-s4-1b0-ternary-20260831/r001/audit

- round: 1
- delivery: asplos27-s4-1b0-ternary-20260831/r001/delivery
- verdict: approve
- from: external-model-auditor -> to: codex-executor
- ts: 2026-08-31T05:10:00Z
- assurance-level-achieved: E2-DIRECT-LEGACY

## Findings

### S4-1B0-AUDIT-F1 [info / closed-verified] 历史9/10演练缺口

- evidence: 外审按披露的攻击面复测：翻动positive-00.bin某coefficient LSB（符号不变）并重绑该行
  binary/index hash、summary sidecar绑定与manifest，被“positive fresh-process binary determinism
  differs”拒绝；4e2a261修复有效，正式artifact生成于修复之后（git祖先顺序核实）
- advice: 无需处理

### S4-1B0-AUDIT-F2 [info / closed-by-disclosure] E0边界

- evidence: 外审自建coherent full resign（伪造abcrown commit+全链重签）被self-check接受；tamper
  report含coherent_full_resign_e0_boundary_disclosed=true，delivery risks已披露；物理真实性由AC1外部
  仓库核验+外审fresh run支撑
- advice: 保持披露；S4-4 challenge+witness结构性覆盖

### S4-1B0-AUDIT-F3 [info / open] 环境限制

- evidence: dol CLI不在外审环境PATH，dol lint/exchange validate未独立复跑
- advice: 由executor侧记录承担

## Summary

S4-1B0 ternary endpoint r001独立外审结论approve：blocker/major/minor均为0，info 3。AC1—AC7逐项
PASS。AC1：7/7 code blob、三外部commit、model/property SHA256、四份合同JSON绑定、manifest 15文件
与外审独立重算一致，无本机路径，正式artifact确在4e2a261修复后生成；AC2：亲读TIR确认exponent-bit
classifier（无x==x）、±0归zero、f32两步舍入midpoint、canonical qNaN 0x7fc00000，PyTorch/bit双独立
oracle无TIR自比，20个stable reason与冻结负向合同逐字一致；AC3：外审从5份.bin原始字节独立重算
selector计数8689/9137/606/0与18,432个selected逐元素bitwise exact，5份sidecar逐字节一致
（a07aea90...）；AC4：cache真实miss→hit且只编译一次，5个fault由独立fresh PID真实触发、reason按
冻结顺序、context none、reject-before-launch、fallback/eager/native-shadow=0；AC5：stdlib replay从raw
语义重算（非只验hash）双份PASS，10/10注册tamper拒绝，外审LSB攻击被新determinism回绑拒绝，
coherent full resign在E0接受且已披露；AC6：targeted 22 passed、全量2073 passed/3 skipped现场复现，
Black/mypy/pylint/diff-check全部真实通过（7文件pylint逐文件10.00/10）；AC7：claim边界一致，无
VALIDATED升级或性能越界。外审亲启11-process fresh generation达到E2-DIRECT-LEGACY（fresh与formal
除source_revision外全字段一致）。同意关闭VALIDATED-S4-1B0-TERNARY-ENDPOINT并只开放S4-1B
production implementation/correctness；S4-4仍须challenge+witness。详见r001/audit_report_full.md。
