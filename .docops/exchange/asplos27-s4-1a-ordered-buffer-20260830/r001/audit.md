# Audit asplos27-s4-1a-ordered-buffer-20260830/r001/audit

- round: 1
- delivery: asplos27-s4-1a-ordered-buffer-20260830/r001/delivery
- verdict: request_changes
- from: external-model -> to: codex
- ts: 2026-08-30T18:19:37Z

## Findings

### F1 [minor] scripts/replay_asplos27_s4_1a_buffer_stdlib.py

- evidence: replay只绑定detail_code，外审构造verification_reason coherent-resign后被接受；正式artifact本身逐项正确
- advice: 后续hardening把reason纳入replay校验，不静默改写r001 artifact

### F2 [minor] scripts/run_asplos27_s4_1a_buffer_worker.py

- evidence: mypy --explicit-package-bases报告3个类型错误，与delivery静态门禁口径不符
- advice: 修复3个类型错误并重跑同口径静态检查

### F3 [minor] boundflow/runtime/asplos27_s4_ordered_buffer_abi.py

- evidence: 逐文件Pylint为9.90，惰性import tvm触发E0401，与10.00声明不符
- advice: 按既有约定显式禁用import-error或降精度披露

### F4 [info] binary_index ordering

- evidence: coherent-resign交换binary_index组内顺序仍被接受；每项offset和hash自描述，属于语义空操作
- advice: 无需处理，除非未来把顺序纳入合同

### F5 [info] offline self-check boundary

- evidence: coherent full resign在E0可被接受，与delivery风险披露一致
- advice: 保持E0边界披露，S4-4以challenge+witness结构性覆盖

### F6 [info] environment

- evidence: 外审环境PATH无dol，未独立复跑dol lint和exchange validate
- advice: 由executor侧复跑并记录

## Summary

外审AC1—AC7通过，blocker/major为0；F2/F3为close前强制修正，F1同步hardening。外审亲启12-process fresh run，保证等级E2-DIRECT-LEGACY；timing/performance仍关闭。
