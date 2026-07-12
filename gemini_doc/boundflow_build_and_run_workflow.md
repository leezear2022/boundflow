# BoundFlow 日常构建、运行与验证工作流

## 1. 进入与退出环境

```bash
cd /home/lee/Codes/boundflow
conda activate boundflow
```

Conda hook 会自动加载 `env.sh`。退出时：

```bash
conda deactivate
```

不要在 hook 已安装后重复手工 source `env.sh`；不要裸 shell 使用系统 `/opt/cuda`，当前系统
toolkit 是 13.3，而论文环境的 Conda toolkit 是 13.2。

## 2. 按源码类型选择动作

| 修改位置 | 是否编译 | 最小动作 |
|---|---:|---|
| `boundflow/*.py` | 否 | 重启 Python，运行相关 pytest |
| `boundflow/3rdparty/tvm/python/tvm/*.py` | 否 | 重启 Python，运行 TVM/相关 pytest |
| `boundflow/3rdparty/tvm/src`、`include`、CMake/TIR C++ | 是 | `bash scripts/rebuild_tvm.sh` |
| TVM 内嵌 `3rdparty/tvm-ffi` Python | 通常否 | 重启 Python；若 extension/ABI 变化按下一行 |
| TVM 内嵌 `3rdparty/tvm-ffi` C++/Cython/ABI | 是 | `bash scripts/install_dev.sh tvm` |
| `environment.yaml` / wheel 版本 | 是 | 执行对应 staged installer，不默认跑 `all` |
| LLVM/MLIR 调用侧的 TVM C++ | 是 | `bash scripts/rebuild_tvm.sh` |

当前唯一有效的 tvm-ffi 来源是：

```text
boundflow/3rdparty/tvm/3rdparty/tvm-ffi
```

顶层 `boundflow/3rdparty/tvm-ffi` 不进入 Python 路径和构建链。

## 3. TVM 增量编译

```bash
bash scripts/rebuild_tvm.sh
python scripts/smoke_tvm_cuda.py
python -c "import tvm, triton; print(tvm.__version__)"
python -m pytest -q <相关测试>
```

构建目录固定为：

```text
boundflow/3rdparty/tvm/build-boundflow
```

禁止执行：

```bash
pip install -e boundflow/3rdparty/tvm
```

TVM 根目录的 scikit-build 会用默认参数重新配置构建目录，可能覆盖 CUDA 13.2、LLVM 静态链接
和 `HIDE_PRIVATE_SYMBOLS` 配置。Python 包通过 activation hook 的 `PYTHONPATH` 加载。

## 4. tvm-ffi ABI 修改

```bash
BOUNDFLOW_BUILD_JOBS=8 bash scripts/install_dev.sh tvm
python -c "import tvm_ffi, tvm, triton; print(tvm_ffi.__version__, tvm.__version__)"
python scripts/smoke_tvm_cuda.py
```

该阶段会使用 TVM 锁定的内嵌 tvm-ffi commit，并重新安装其 Python extension。不要复制 `.so`
到另一个源码树。

## 5. LLVM/MLIR 边界

当前环境使用 Conda LLVM/Clang 20.1.8，仓库没有独立 LLVM/MLIR 源码树。修改 TVM 中调用
LLVM 的代码只需重编 TVM。若未来必须修改 LLVM/MLIR 本体，应安装到独立前缀并调整
`scripts/llvm-config-static.sh`，不得覆盖当前 Conda LLVM；启用 TVM `USE_MLIR` 也必须作为
独立构建配置和变更记录，不能假设当前已经启用。

## 6. 分层验证

Python-only 修改：

```bash
python -m pytest -q tests/<相关测试>.py
```

TVM/CUDA 修改：

```bash
bash scripts/rebuild_tvm.sh
python scripts/smoke_tvm_cuda.py
python -c "import tvm, triton"
python -m pytest -q tests/<相关测试>.py
```

合并或提交前：

```bash
bash -n scripts/*.sh
git diff --check
python -m pytest -q tests
```

环境级严格验证：

```bash
bash scripts/install_dev.sh verify
```

Gate 0 的 MLP/CNN 稳定性基线：

```bash
bash scripts/install_dev.sh baseline
```

该阶段使用 `run_phase5d_artifact.py --mode reduced`，每个 workload 做 3 次 warmup 和 10 次
计时，只覆盖 small matrix。它用于发现环境迁移前后的明显回归，不等价于论文要求的至少 5 次
独立重复；论文数据仍需用独立 run ID、固定 workload/config 和完整统计流程采集。

## 7. ASPLOS 证据要求

研究 PR 还必须生成或更新：

- `gemini_doc/change_YYYY-MM-DD_*.md` 与 `docs/change_log.md`；
- 相关 correctness/gradient tests；
- 原始 JSONL（包括 OOM/timeout/error）；
- schema validation、CSV、table/figure、manifest；
- `gemini_doc/asplos_claims_map.md` 中对应 claim 的状态。

详细门禁见 `gemini_doc/asplos_execution_memo_v1_0.md`。
