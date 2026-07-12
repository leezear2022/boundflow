# 变更记录：CachyOS / PyTorch 2.12.1 / CUDA 13.2 环境链路

## 目标与边界

本轮先修复环境启动链，不推进新的 Planner 研究功能。没有运行旧版
`scripts/install_dev.sh`，没有使用 sudo，也没有修改驱动、内核或系统 `/opt/cuda`。

## 现场审计

- 主机：CachyOS，Linux 7.1.3；RTX 4060 Laptop GPU；driver 610.43.03。
- 系统工具链已滚动到 CUDA 13.3、LLVM/Clang 22.1.8，不能作为论文目标环境。
- Conda 位于 `/home/lee/miniconda3`，开始时不存在 `boundflow` 环境。
- TVM commit `6248b5d` 内嵌 tvm-ffi commit `ae346ec`；仓库顶层 tvm-ffi
  是较新的 `438f643`。旧脚本会让 C++ 与 Python 分别来自两个 commit。
- auto_LiRPA commit `9d100ec` 声明 `torch<2.9`，因此只能 `--no-deps` 安装。

## 修改

- `environment.yaml`：锁 Python 3.12、LLVM/Clang 20.1.8、CUDA toolkit 13.2，
  增加 PyTorch 2.12 ONNX exporter 所需 `onnxscript` 和 `python-graphviz`。
- `requirements-pytorch-cu132.txt`：只从 PyTorch 官方 cu132 index 安装
  `torch==2.12.1`、`torchvision==0.27.1`。
- `scripts/install_dev.sh`：改成显式阶段入口；TVM 使用专用
  `build-boundflow/`、Clang 20.1.8、nvcc 13.2、`HIDE_PRIVATE_SYMBOLS=ON`。
- installer 的 `conda run` 路径统一静默加载 hook，避免 hook 提示污染 CUDA/Clang 路径探测；
  `verify` 也不再要求调用者预先激活 Conda 环境。
- `scripts/llvm-config-static.sh`：让 TVM 静态链接 LLVM，再隐藏符号。仅动态链接
  libLLVM 时，`import tvm; import triton` 会以 `free(): invalid pointer` abort。
- TVM 与 Python 统一使用 TVM 内嵌 tvm-ffi；不再编译/复制顶层 fork 的 `.so`。
- 新增 host/环境 doctor、nvcc smoke、TVM TIR CUDA smoke。
- `crown_ibp.py` 增加语义保持的 reshape 前向/反传，以兼容 PyTorch 2.12
  ONNX exporter 为 residual/concat 图新增的 Reshape。

## 验证结果

- PyTorch `2.12.1+cu132`、torchvision `0.27.1+cu132`、CUDA runtime 13.2；
  `torch.cuda.is_available()` 为真。
- Conda nvcc 13.2.78 最小 C++20 CUDA kernel：通过。
- LLVM/Clang 20.1.8；TVM 0.23.dev0，CUDA + LLVM 构建：通过。
- `ldd libtvm.so` 不再依赖共享 libLLVM；`import tvm; import triton`：通过。
- TVM TIR CUDA 实际执行并校验输出：通过。
- auto_LiRPA toy MLP/CNN IBP 与 BoundFlow CROWN 组合门禁：9 passed。
- 全量：`162 passed, 1 skipped`；skip 是 TVM 可用时主动跳过 no-TVM 重复编译。
- quick baseline：MLP 与 MNIST CNN 各一条 schema 1.0 JSONL，均满足
  Python↔TVM、Python↔auto_LiRPA correctness gate。产物位于
  `artifacts/environment-baseline/env-cu132-20260712/`（按仓库规则不入 Git）。

## 已知风险

- `cuda-toolkit=13.2` 完整 Conda 包体积较大；系统 `/opt/cuda` 仍是 13.3，必须
  使用脚本或激活环境，避免裸 shell 误选系统 nvcc。
- CUDA 13.2 headers 在 Clang 严格告警下产生 reserved macro 等 warning，不影响构建。
- PyTorch 2.12 的 ONNX exporter 会将请求 opset 17 先按 18 导出再转换，测试已通过，
  但长期应显式评估并冻结目标 opset。
- quick baseline 只有单次计时，只用于链路验收，不用于论文性能结论。

## Go / No-Go

环境链路为 **Go**。可以进入下一个研究型 PR，但应先把本环境 manifest 与更稳定的
多次 baseline 封箱；不要把本次 quick 数字作为 operator-preserving CROWN 的收益基线。
