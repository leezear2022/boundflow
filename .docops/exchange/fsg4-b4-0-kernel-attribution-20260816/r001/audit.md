# Audit fsg4-b4-0-kernel-attribution-20260816/r001/audit

- round: 1
- delivery: fsg4-b4-0-kernel-attribution-20260816/r001/delivery
- verdict: approve
- from: external-model -> to: codex
- ts: 2026-08-15T23:15:52Z

## Findings

### F1 [minor] boundflow/runtime/fsg4_b4_kernel_attribution.py:135-145,378-399

- evidence: CUDA kernel 行 input_shapes 全空(0/35367),形状仅在 cpu_op 行(182813/235201);为 torch profiler 固有行为
- advice: B4-B 冻结 production shape 时显式记录 kernel 形状从 correlation parent operator 行恢复

### F2 [minor] .docops/exchange/fsg4-b4-0-kernel-attribution-20260816/request.md:62

- evidence: B3/B4相关=54 passed 的文件集合未固定;审计以 same_solver×3+explicit_counters+b4×2 复现恰为 54,全并集为 96 也全绿
- advice: 后续 exchange 在 request 固定 related 文件清单

### F3 [info] gemini_doc/change_2026-08-16_fsg4_b4_0_kernel_attribution_closure.md:46-51

- evidence: 67.72% 是 span 级 wall share;14 次 CROWN call kernel-sum 占该 region kernel-sum 的 68.3%(32.62/47.75ms),region 含 forward/KFSB score 等非 CROWN 工作;文档已披露换算来源,5% 门槛在保守口径下不变
- advice: 保持披露口径即可

### F4 [info] events/profile.jsonl.gz

- evidence: kfsb.crown.01/02 的 CUDA 时间戳早于 optimizer.crown.00,系 CPU/CUDA 时钟域混合;ordinal 归属全基于显式 marker,不受影响
- advice: 无需处理

## Summary

外审从 formal raw 用标准库独立重算,AC1—AC7 全部 PASS。hash 链(manifest 13 文件/manifest_hash/protocol_hash/worker_hash/解压 JSONL/canonical raw)全部自算一致;source 恰为 66154e4,10 个 code blob、B3 manifest、模型/property、三个外部仓库 commit 逐一核对一致;artifact 零本机路径泄漏。control/profile 为顺序 fresh B3 worker,discrete/sign exact,lower max abs diff=4.768e-07≤2e-4。独立解析 270,609 events:35,367 kernel 全归因 0 丢失,correlation 33,060/temporal 2,307 与 raw 一致;14 CROWN+4 forward ordinal 精确重建;41 条 CUDA user annotation 未误算;2,307 temporal fallback 显式标记且 containment 0 违规。opportunity ledger 独立复算逐项吻合(CROWN14=9,196 kernel/32,618,329ns/3,291 mat ops/57,292,800B);67.72% 换算核到 B3 formal raw 逐位一致,required_r=3.9897x。B4-A 重复 call 有 raw 结构证据(terminal 36 kernel names=optimizer.crown.09);B4-B≥5% 成立。replay 逐字节一致;tamper 9/9 拒绝,审计方自建第 10 类全重签变体仍被语义重算拒绝。测试 15/54/1329+3skipped、black/mypy/pylint 10.00/diff/dol validate×2/lint 全过;origin 与本地一致。claim 边界无漂移:无 speedup/B0-parity/memory/ASPLOS-ready 表述。同意关闭 VALIDATED-B4-0-OPPORTUNITY 并只开放 B4-A;B4-B 可设计不得合并执行,B4-C/D 与 B5—B7 继续关闭。详见 r001/audit_report_full.md。
