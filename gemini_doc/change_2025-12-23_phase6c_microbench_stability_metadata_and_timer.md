# 变更记录：Phase 6C（CROWN-IBP multi-spec）microbench 稳态增强——元信息/计时后端/串行口径说明

## 动机

Phase 6C 的 microbench 目标是量化 “multi-spec 真 batch vs 串行逐 spec” 的吞吐收益。为避免 reviewer/复现实验常见质疑点，本次把 bench 口径进一步“工程化钉死”：

- 明确 warmup/repeat 次数，并把它们写入 stdout 的 JSON payload，便于复现实验与回归对比；
- 提供可选计时后端（`torch.utils.benchmark`），让稳态计时更可控；
- 显式说明串行 baseline 包含 Python 循环开销（符合真实 per-spec 串行用法，但需要口径透明）；
- forward 复用回归在 `inference_mode` 下运行，避免 autograd 上下文影响路径/开销。

## 本次改动

- 更新：`scripts/bench_phase6c_crown_ibp_multispec_throughput.py`
  - 参数增强：
    - 默认 `--warmup=10 --iters=50`（稳态更稳定）；
    - 新增 `--serial-warmup/--serial-iters`（默认跟随 batch 参数）；
    - 新增 `--timer {perf_counter,torch_benchmark}` 与 `--torch-benchmark-min-run-time-s`（可选使用 `blocked_autorange`）。
  - 输出增强：
    - stdout JSON 增加 `meta`：`device/dtype/B/shape/eps/p/specs_list/warmup/iters/serial_* / torch_version / timer` 等；
    - stderr 输出一条口径说明：`serial_ms_*` 包含 Python loop + slicing overhead。
  - 计时增强：
    - 计时路径统一使用 `torch.inference_mode()`；
    - CUDA 场景下在 timed 部分同步（避免异步计时偏差）。

- 更新：`tests/test_phase6c_crown_ibp_multispec_batch.py`
  - forward 复用回归用例包裹 `torch.inference_mode()`，降低上下文差异引入的波动风险。
- 更新：`docs/change_log.md`
  - 追加 Phase 6C microbench 稳态增强的总账条目。

## 如何验证

```bash
python -m pytest -q tests/test_phase6c_crown_ibp_multispec_batch.py

# 默认计时（perf_counter）
python scripts/bench_phase6c_crown_ibp_multispec_throughput.py --device cpu --specs-list 1,4,16,64

# 可选：torch benchmark 计时后端
python scripts/bench_phase6c_crown_ibp_multispec_throughput.py --device cpu --timer torch_benchmark --specs-list 1,4,16,64
```
