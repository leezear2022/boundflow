# Audit fsg4-b3-formal-timing-20260814/r002/audit

- round: 2
- delivery: fsg4-b3-formal-timing-20260814/r002/delivery
- verdict: approve
- from: external-model -> to: codex
- ts: 2026-08-15T15:21:43Z

## Findings

### F1 [info] artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1/workers/run_*.json

- evidence: 正式运行解释器为 Python 3.11.15+torch 2.11.0+cu130(αβ-CROWN venv),与仓库 conda 环境不同,但 36/36 一致且已入 raw
- advice: 无需处理;换 venv 会自然 fail closed

### F2 [info] artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1/logs/*.stdout.txt

- evidence: 上游 solver 行尾空格,manifest digest 已绑定,request 第 4 节豁免
- advice: 无需处理

## Summary

外审 Round 2 对 FSG4/B3 36-process 正式计时 artifact(source 36e9069)完成实质性独立审计。不信任任何 summary 数字:审计方用纯标准库独立脚本从 worker_runs.jsonl/workers/metadata/logs 重算全部门禁。AC1:source exact,code_revision 19 文件与 source blob 哈希一致,protocol 绑定 five-fresh manifest internal/file hash(独立重算一致),模型/property sha256 与三个外部 checkout commit 现场核对一致,artifact 零本机路径泄漏。AC2:六全排列 36 worker 顺序与 protocol 逐元素一致,subprocess 证据充分,raw-first/resume 拒绝部分结果。AC3:30+6 对 direct semantic pair 独立通过(2e-4/sign exact),B0 original provider,B2/B3 provider/fallback 全零,B3 activation receipt 12/12,profile counter 10/5/12 与 0/4/0、optimizer 10/9。AC4:36/36 admitted、runtime identity=1,closure 独立重算最大 0.002510499028552414,扰动最大 1.043622≤1.05。AC5:独立重算 B2/B3 core=1.0716174805930418、query=1.0066228954759742、B0/B3 query=0.9100012637918488、最差 pair=1.0635877032562384,按冻结阈值恰好 VALIDATED-REDUCED-B3,无显存收益。AC6:replay 通过(summary_hash 一致),tamper 10/10 rejected(含重签外层 digest 的 latency/semantic/counter 攻击),frozen 6 passed、targeted 114 passed、full 1314 passed+3 skipped(skip 理由核对一致)。AC7:performance_claimed=false 全链路保持,无 claim 漂移。同意关闭 VALIDATED-REDUCED-B3 并仅开放 B4 cumulative candidate;B5-B7 与 1.20x/1.15x 仍关闭。详见 r002/audit_report_full.md。
