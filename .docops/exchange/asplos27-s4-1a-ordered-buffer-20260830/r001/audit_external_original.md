# Audit asplos27-s4-1a-ordered-buffer-20260830/r001/audit

- round: 1
- delivery: asplos27-s4-1a-ordered-buffer-20260830/r001/delivery
- verdict: approve-with-minor-correction
- from: external-model-auditor -> to: codex-executor
- ts: 2026-08-31T02:30:00Z
- assurance-level-achieved: E2-DIRECT-LEGACY

## Findings

### S4-1A-AUDIT-F1 [minor / open] scripts/replay_asplos27_s4_1a_buffer_stdlib.py

- evidence: 冻结合同把7个fault映射到(detail_code, verification_reason)二元组，但stdlib replay只校验
  detail_code；外审自建攻击把fault `output`的reason从UNSAFE_ALIAS_OR_LIFETIME伪造成
  RECEIPT_IDENTITY_MISMATCH并coherent重签，replay接受。当前正式artifact的7个reason经外审人工比对
  与冻结表逐项一致，77项negative测试在代码层同时断言detail+reason，运行时无实际缺口
- advice: 后续hardening提交把reason纳入replay校验；不得在已交付r001上静默补丁；不阻塞本轮批准

### S4-1A-AUDIT-F2 [minor / mandatory-before-close] scripts/run_asplos27_s4_1a_buffer_worker.py

- evidence: 按S3既有口径`mypy --explicit-package-bases`对7个交付文件检查，worker脚本有3个类型错误
  （L245 None赋给Tensor、L297/L301 monkey-patch类赋值）；类型级问题，无运行时影响（formal与fresh均
  实际跑通），但与delivery“black-mypy-pylint pass”口径不符
- advice: 修复3个错误并重跑静态门禁，或在权威文档把静态门禁口径降精度为实际结果；artifact replay不
  绑定git blob，代码修复不影响已交付artifact

### S4-1A-AUDIT-F3 [minor / mandatory-before-close] boundflow/runtime/asplos27_s4_ordered_buffer_abi.py

- evidence: 逐文件pylint该文件为9.90/10（E0401无法解析惰性import tvm），与“Pylint 10.00”声明不符；
  S2既有约定是文件头禁用import-error，本文件未遵循；其余6个交付文件均为10.00/10
- advice: 补`# pylint: disable=import-error`头注释或降精度披露；不阻塞正确性结论

### S4-1A-AUDIT-F4 [info / closed] binary_index顺序

- evidence: 外审自建binary_index组内顺序交换+coherent重签被接受；每项自带offset/hash，顺序不是合同
  语义，属语义空操作
- advice: 无需处理；如未来把index顺序纳入合同，需在replay显式校验

### S4-1A-AUDIT-F5 [info / closed-by-disclosure] E0边界

- evidence: coherent full resign（伪造abcrown commit+全链重签）被self-check接受，与delivery risks
  披露一致；offline self-check只证明E0闭包，物理真实性由AC1外部仓库核验+外审fresh run支撑
- advice: 保持披露；S4-4 challenge+witness结构性覆盖

### S4-1A-AUDIT-F6 [info / open] 环境限制

- evidence: dol CLI不在外审环境PATH，dol lint/exchange validate未独立复跑
- advice: 由executor侧记录承担

## Summary

S4-1A ordered buffer r001独立外审结论approve-with-minor-correction：blocker 0、major 0、minor 3、
info 3。AC1—AC7逐项PASS：AC1 9/9 code blob、三外部仓库commit、model/property SHA256、manifest与
全hash链外审独立重算一致，无本机路径，scope确认只有buffer prepare；AC2亲读确认S4-0 ticket单次消费、
16 buffer冻结顺序（6α→1β→6dα→1dβ→lower→upstream）、16互异storage、三方alias互斥、完整view key、
noncontiguous前置拒绝、fault路径无exception context保留；AC3外审从raw+.bin独立重算：40/40
source/candidate逐字节相等（formal与fresh双份）、parameter/gradient=4254/17016、storage/view=16/16、
candidate logical=34080、empty β=5/0、D2H=32/85056、累计56/153072、D2D=7/17016、allocated delta
5/5为0，且10个进程的5个sidecar逐字节全同（78970266...）；AC4 7个隔离故障各自独立fresh进程、
detail/reason与冻结表一致、context none、delta=0，外审fresh生成重跑了全部7个；AC5 registry 77唯一
nodeid（minimum 68）、unit 80、artifact+unit 84、stdlib replay（无boundflow/torch/tvm/numpy import）
双份PASS、10/10 tamper语义级拒绝，外审另建三类攻击（一类语义空操作、一类揭露F1、一类确认E0
边界）；AC6全量2050 passed/3 skipped与声明一致，black/diff-check通过，mypy/pylint两处口径差异见
F2/F3；AC7 claim边界一致无越界。外审计亲启12-process fresh run达到E2-DIRECT-LEGACY。同意关闭
VALIDATED-S4-1A-ORDERED-BUFFER-PREPARE并只开放S4-1B0 implementation/correctness；F2/F3修正落实
前exchange不标记approved；S4-4仍须challenge+witness。详见r001/audit_report_full.md。
