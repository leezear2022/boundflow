# Audit fsg4-b4b2-b2-4-sparse-conv-20260823/r001/audit

- round: 1
- delivery: fsg4-b4b2-b2-4-sparse-conv-20260823/r001/delivery
- verdict: approve
- from: external-model -> to: codex
- ts: 2026-08-23T16:02:22Z

## Findings

### F1 [minor] gemini_doc/BOUNDFLOW_FSG4_B4B2_B2_4_SPARSE_CONV_TIR_CHANGELOG_2026_08_23.md:69

- evidence: 候选5 schedule hash 缩写写作 6ab7c314…1646,真实 64 位 hash 末4位为 646d(6ab7c314f51c7f4e193e777568e2730a51d3aa501e5aeb2d161d543a05a1646d);ledger 完整 hash 与独立重算逐位一致,真实 hash 集合无误,仅文档表格后缀截取笔误
- advice: 修正表格后缀为 …646d,不影响门禁与 hash 链

### F2 [info] .docops/ev.jsonl

- evidence: dol validate 报 3 个 duplicate event id(ev009180/ev009388/ev010862),在 base commit b18fad4 即存在(2026-08-14 及 08-23 凌晨历史事件),非本轮引入;dol lint --soft 通过
- advice: 后续窗口期清理历史重复 id,本轮不阻塞

### F3 [info] boundflow/runtime/fsg4_b4b2_sparse_conv_tir.py

- evidence: module TIR/device-source hash 在 validate 仅做格式校验,独立重编译比对按预注册明确留待 B2-5 replay(B2-3 遗留 info 延续,交接如实声明未虚假关闭)
- advice: B2-5 列为硬性验收项

## Summary

B2-4 审计全部现场复核通过,AC1—AC8 全部 PASS。AC1:1f8d47a 恰在已关闭的 b18fad4 之后,diff 仅 6 个声明文件+文档,预注册门禁条文(knob 集合/≤12 冻结)对比 b18fad4 逐字未动。AC2:直解 5 份 raw capture,production α[2,1,6,86]、β/location/sign 均 [6,0],86 坐标唯一且在域内;ABI 9 输入无任何 β/native-α buffer。AC3:审计方用 numpy float64 闭合公式独立实现 forward/backward oracle(非 repo reference),5 raw×4 路输出与 GPU TIR 最大差 1.83e-06、sign exact;raw native α grad 在 516 owned 外严格为零。AC4:现场 runner 复现 5/20/64050、max diff 2.384185791015625e-06、cache miss+hit×4、template/P0 module hash 逐位一致。AC5:12 候选 knob 为预注册集合 balanced subset(逐轴全覆盖、无越界),12 schedule/module hash 唯一、cache 全 miss,ledger hash 1660edca…07c6 独立重算逐位一致;五个冻结字段由 validate 代码强制;审计方亲手做 12 项篡改探针(越界 knob/第13候选/乱序/翻转冻结字段/篡改 hash/重签)全部 fail-closed;无 timing/winner/排序代码。AC6:12 个 scheduled TIR alloc_buffer 恰为 adjoint_conv[6,1,16,8,8]+output_bias_delta[6,1],无 dense α/β/scaled-A/scatter workspace。AC7:B2-3 遗留 shape-mismatch 用例已补;hash 重编译比对如实留 B2-5。AC8:targeted 51、related 105、full 1465 passed/3 skipped(skip 均为既有环境边界)、black/mypy/pylint 10.00/ninja no work/dol 全过。claim 边界五处文档一致,无漂移。结论 APPROVE,同意关闭 B2-4 并仅开放 B2-5(formal independent-process artifact/replay/AB-BA timing,必须复用冻结 12 项 ledger,不得追加第 13 候选;B2-5 验收必须含 module TIR/device-source hash 独立重编译比对)。详见 r001/audit_report_full.md。
