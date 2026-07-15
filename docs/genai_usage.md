# Generative AI Usage Log

本文件记录生成式 AI 对 BoundFlow 文字、代码、测试和实验工件的协助范围。所有接收内容均由
项目维护者通过代码审阅、自动测试和工件检查验证；AI 输出不被视为独立正确性证据。

| 日期 | 工具/模型 | 任务 | 生成或修改内容 | 人工/自动验证 | 接收边界 |
|---|---|---|---|---|---|
| 2026-07-14 | OpenAI Codex | PR-13A query runtime foundation | `BoundQuery`/state-validity/recorder/replay、BaB observer、测试、runner 与研发文档 | focused pytest、固定流 replay、Mypy/Pylint/全量测试（提交前补齐） | 接收 contract/replay foundation；不接收性能、non-toy 或 full solver claim |
| 2026-07-14 | OpenAI Codex | PR-13B dynamic batching foundation | BatchManager、physical αβ pack/unpack、deadline/budget/OOM tests、artifact 与文档 | deterministic unit tests、8-query dynamic/fault replay、static checks | 接收 correctness/mechanism；不接收 CPU clock、fault OOM 为性能/真实 GPU 证据 |
| 2026-07-14 | OpenAI Codex | PR-13C same-solver adapter | optional solver adapter、rich αβ result state、capability non-invocation test、artifact 与文档 | original/runtime query/state/search-counter comparison | 接收 same-solver correctness foundation；不接收单次 wall time 为 speedup |
| 2026-07-14 | OpenAI Codex | PR-13D/E reduced GPU 与 closure | GPU benchmark、stream/cache/hot-path 修正、closure audit、Artifact Appendix 与 claims 更新 | RTX 4060 5-repeat fixed/E2E、custom-stream test、focused/full tests、静态与污染审计 | 接收 `VALIDATED-REDUCED`；拒绝把 batching 收益写成 runtime 独立贡献，拒绝 non-toy/compiled-Planner/full-C3 claim |

后续每个研究切片继续追加日期、工具、任务、产物、验证方式和被拒绝/未完成范围。
