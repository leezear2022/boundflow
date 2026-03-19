# Quick Restart：像 auto_LiRPA 一样跑 IBP 边界（BoundFlow）

目标：给定 **PyTorch 模型 + 输入中心 `x0` + `L∞` 扰动半径 `eps`**，快速得到输出的区间边界 `lb/ub`；并可选对齐 `auto_LiRPA` baseline、或走 TVM executor 做性能路径。

---

## 0) 环境启动（只做一次/或重启后做）

```bash
conda activate boundflow
source env.sh
```

可选自检（推荐）：  
```bash
pytest -q tests/test_env.py
```

如果是新机器/新环境，按仓库约定用：  
```bash
bash scripts/install_dev.sh
```

---

## 1) 最快路径：直接用 bench 脚本出 JSONL（推荐）

### 1.1 Python-only（不依赖 TVM）

```bash
python scripts/bench_ablation_matrix.py \\
  --workload mlp --matrix small --batch 1 --eps 0.1 \\
  --warmup 1 --iters 3 \\
  --no-tvm \\
  --output out/quick_ibp.jsonl
```

说明：
- 不传 `--output` 时会写到 stdout（可用 `> out.jsonl` 重定向）。
- 默认会尝试跑 `auto_LiRPA` baseline；如果环境里没有或你想更快：加 `--no-auto-lirpa`。
- JSONL 字段口径见 `docs/bench_jsonl_schema.md`。

### 1.2 Python + TVM（首次会编译，后续走 cache）

```bash
python scripts/bench_ablation_matrix.py \\
  --workload mlp --matrix small --batch 1 --eps 0.1 \\
  --warmup 1 --iters 3 \\
  --tvm-cache-dir .benchmarks/tvm_cache \\
  --output out/quick_ibp_tvm.jsonl
```

---

## 2) 一键产物：artifact runner（面向论文/AE）

```bash
python scripts/run_phase5d_artifact.py --mode quick --workload mlp
```

如果本机没有 TVM（或不想覆盖 TVM 相关 claim），允许生成 python-only 产物：
```bash
python scripts/run_phase5d_artifact.py --mode quick --workload mlp --allow-no-tvm
```

---

## 3) “像 auto_LiRPA compute_bounds” 的最小 Python API 示例

这个例子走 reference 路径：`import_torch` → `PythonInterpreter.run_ibp()`，直接得到 `IntervalState(lower, upper)`。

```bash
python - <<'PY'
import torch

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.runtime import LinfInputSpec, PythonInterpreter


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)

    def forward(self, x):
        return torch.relu(self.fc(x))


model = M().eval()
x0 = torch.randn(1, 10)
eps = 0.1

program = import_torch(model, (x0,))
input_name = program.graph.inputs[0]

state = PythonInterpreter().run_ibp(
    program,
    LinfInputSpec(value_name=input_name, center=x0, eps=eps),
)
print("lb shape:", tuple(state.lower.shape))
print("ub shape:", tuple(state.upper.shape))
print("lb[0,:3] :", state.lower[0, :3])
print("ub[0,:3] :", state.upper[0, :3])
PY
```

---

## 4) 常见问题（restart 时最常见的坑）

- `auto_LiRPA` 不可用：bench 加 `--no-auto-lirpa`；或重跑 `bash scripts/install_dev.sh`（会以 editable 方式装 vendored 的 `boundflow/3rdparty/auto_LiRPA`）。
- TVM 不可用：bench 加 `--no-tvm`；artifact runner 加 `--allow-no-tvm`。
- 输出不止一个 value：`PythonTaskExecutor.run_ibp(..., output_value=...)` / `run_ibp_scheduled(..., output_value=...)` 需要显式指定输出 value 名。

