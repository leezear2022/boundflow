# 变更记录：Phase 6C（CROWN-IBP MLP）multi-spec 真 batch——吞吐 microbench + forward 复用回归

## 动机

Phase 6B 已将 CROWN-IBP（MLP: Linear+ReLU）的 correctness 风险用测试与结构 gate 钉死；Phase 6C 需要开始验证“系统收益”：

- multi-spec（`C:[B,S,O]`）一次性批处理相对串行逐 spec 的吞吐提升；
- forward IBP 的 ReLU pre-activation bounds 与 spec 维度解耦，应当 **只算一次**（不随 S 增加而重复）。

## 本次改动

- 新增：`scripts/bench_phase6c_crown_ibp_multispec_throughput.py`
  - 对同一 MLP/输入/扰动，比较 `S={1,4,16,64}`：
    - `batch_ms_p50`：一次性 multi-spec batch 调用 `run_crown_ibp_mlp(..., C:[B,S,O])`；
    - `serial_ms_p50`：循环 `S` 次 `C[:,s:s+1,:]` 的串行总耗时；
    - `speedup = serial / batch`。
  - stdout 输出 JSON payload；进度与解释写 stderr（便于管道/重定向）。

- 新增：`tests/test_phase6c_crown_ibp_multispec_batch.py`
  - `test_phase6c_forward_ibp_work_independent_of_specs`：通过 monkeypatch 计数 `IntervalDomain.{affine_transformer,relu_transformer}` 的 forward 调用次数，断言在 `S=1` 与 `S=32` 下次数相同，从而验证 forward IBP 不随 spec 维度重复计算（Phase 6C 的核心复用点）。
- 更新：`docs/change_log.md`
  - 追加 Phase 6C 总账条目（本 microbench + forward 复用回归）。

## 如何验证

```bash
python -m pytest -q tests/test_phase6c_crown_ibp_multispec_batch.py
python scripts/bench_phase6c_crown_ibp_multispec_throughput.py --device cpu --specs-list 1,4,16,64
```

## 备注

- microbench 的计时口径采用 p50，并在 CUDA 上使用 `torch.cuda.synchronize()` 以避免异步计时偏差。
- forward 复用测试只统计 forward IBP 的 domain transformer 调用次数；backward（随 S 扩展）不纳入计数。
