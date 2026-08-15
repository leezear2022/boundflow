# FSG4/B4 cumulative CUDA/TIR fusion changelog

## 2026-08-16 — v1 preregistration

- 以 externally approved B3为直接基线，B0为累计公平对照；
- 从 B3 raw独立重算 core/query share 与 required-speedup公式；
- 冻结14次 production lower-only CROWN调用的全覆盖目标，而非 optimizer-only局部优化；
- 将 B4拆为 B4-0 attribution、B4-A terminal export fusion、B4-B differentiable lower-only TIR、
  B4-C cumulative coverage与B4-D formal timing；
- 明确 PR-12 plain-CROWN TIR不具备grad/α/β/split capability，禁止静默复用；
- 冻结 semantic/autograd/physical activation/replay/tamper与B3/B0双基线门禁；
- B5 JIT/CUDA Graph、B6 runtime、B7 memory继续关闭；
- 当前状态为`PREREGISTERED-NOT-IMPLEMENTED`，下一唯一动作是B4-0。
