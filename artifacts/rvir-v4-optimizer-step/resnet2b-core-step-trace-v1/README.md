# RVIR-v4 V4-2B Production Optimizer Step Trace

This artifact binds one production solver core to ten nested bound evaluations, nine observed Adam steps, the two learning-rate schedules, and all 24 alpha/SparseBeta tensors before every evaluation. Replay rebuilds the typed trace from raw tensors and fails closed on semantic tampering. It does not claim optimizer replacement or performance.
