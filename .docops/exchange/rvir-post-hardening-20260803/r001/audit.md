# Audit rvir-post-hardening-20260803/r001/audit

- round: 1
- delivery: rvir-post-hardening-20260803/r001/delivery
- verdict: approve
- from: external-model -> to: codex
- ts: 2026-08-03T16:47:32Z

## Findings

### F1 [info] scripts/run_phase6h_artifact.sh:28

- evidence: PHASE6H_PYTHON 与 CONDA_PREFIX 均未设置时回退到 PATH 中的 python,仍依赖调用者 PATH;runner 亦依赖 env.sh 注入的 PYTHONPATH(完全 strip 环境后 sweep 无法 import boundflow)。属文档化行为,非缺陷
- advice: 可选:runner 内自检 import boundflow 并给友好报错,不阻塞 approve

## Summary

独立复审确认 RVIR 审计后加固(PR #5-#8)全部闭环,无 claim drift。AC1:四条负向测试断言具体错误,backend mismatch 在 exact call 前经 validate 拒绝,正向恰一次 launch/emit。AC2:受限 PATH+显式解释器可生成 artifact,无 override 选 CONDA_PREFIX/bin/python,缺失解释器 exit 2 fail closed,缺 torch 错误未掩盖。AC3:旧 exchange 零改动,两个 exchange validate 与 lint --soft 均 PASS。AC4:v2 独立解析 377/377、root/parent 30/347、parent 先于 child、全部 completed,digest 独立重算吻合;replay 对 377 条重编译五层 IR 逐行比 hash;tamper 测试在重写 payload 并同步更新 digest 后仍因语义重算失败;v1 artifact 未改可 replay;fused 0/394 保留。AC5:mktemp 独立环境固定 clone 三仓库 commit 全部吻合,onnx/vnnlib SHA256 吻合,fresh CPU 重跑 status=ok,12 冻结字段与 8 个 tensor digest 全部独立复现(max diff 3.0994415283203125e-06,sign 9/9)。AC6:targeted 12 passed、全量 460 passed/37 skipped(skip 全部为 CUDA/环境边界)、black/mypy/pylint(10.00/10)全过;无 performance/GPU/本地 αβ kernel/完整 E2E/ASPLOS-ready 漂移,request §6 八条限制保留,IR-5 仍 VALIDATED-NO-GO。RVIR 保持 VALIDATED-REDUCED。详见 r001/audit_report_full.md。
