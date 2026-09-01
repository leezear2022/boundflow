# 修改记录：MR3 Single-Site Timing 预注册

> 日期：2026-08-26

- 在MR3 correctness closure之后冻结唯一下一实验；
- 固定6 pair/12 fresh `PB/BP/PB/BP/PB/BP`、完整outer exact-call host headline与CUDA-event诊断；
- 明确剥离formal CPU-copy/hash/rollback观测，candidate compile与dummy module warm不计时；
- 冻结`1.05/1.00/0.98x` geomean/bootstrap/worst门禁和absolute memory `<=1.05x`；
- GO也只开放complete-query timing预注册；NO-GO保留correctness并停止性能传播；
- 当前没有采样、没有performance claim。
