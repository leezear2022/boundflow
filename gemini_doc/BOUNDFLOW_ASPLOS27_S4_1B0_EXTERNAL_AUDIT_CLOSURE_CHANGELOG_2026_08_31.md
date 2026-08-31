---
status: validated-s4-1b0-ternary-endpoint
date: 2026-08-31
stage: s04
performance-claimed: false
---

# S4-1B0 Round 1 外审关闭修改记录

## 结论

外审报告已独立核验并通过DocOps正式提交：

- exchange：`asplos27-s4-1b0-ternary-20260831`；
- verdict：`approve`；
- approved round：`1`；
- assurance：`E2-DIRECT-LEGACY`；
- AC1—AC7：全部PASS；
- blocker/major/minor/info：`0/0/0/3`；
- executor `exchange validate`：PASS；
- executor `dol lint --soft`：PASS。

外审原始`audit.md`以SHA256
`293fd445f24cefba685b01424db94798d5afe640a827eb64e8f0850f857009c9`保留为
`r001/audit_external_original.md`；DocOps生成的正式audit与closure另行保存，避免未经状态机登记的文件冒充
已提交审计。

## 三项info处置

1. 历史`9/10` replay缺口：外审以符号不变coefficient LSB攻击复测，新的fresh-process determinism回绑
   成功拒绝，已验证关闭；
2. E0 coherent full-resign：已披露，保持该边界，S4-4强制challenge+witness；
3. 外审环境无`dol`：executor侧已实际运行exchange validate与lint，均PASS。

没有finding需要代码修正或下一轮response。

## 状态与后继

S4-1B0升级为`VALIDATED-S4-1B0-TERNARY-ENDPOINT`。唯一开放的后继是S4-1B六site production
implementation/correctness；timing、performance、same-solver、complete-query、10x、S4-1C/S4-1D/S4-2/
S4-3/S4-4 execution均不会因本次批准自动开放。
