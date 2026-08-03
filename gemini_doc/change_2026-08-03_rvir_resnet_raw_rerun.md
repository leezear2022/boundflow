# RVIR ResNet 原始数值重跑证据

> 日期：2026-08-03
> 基线：`main@6a41439`（PR #7 merge）
> 分支：`docs/rvir-resnet-rerun-evidence`
> 范围：外部审计 F5/M5 后续；CPU correctness only

## 1. 起因与结论

外部审计已同意 RVIR 以 VALIDATED-REDUCED 关闭，但 F5/M5 指出审计现场没有 external
αβ-CROWN 环境，因此只能核对冻结摘要、digest 与生成端门禁，不能重新产生 ResNet 原始数值。

本轮从固定 upstream commit 和固定 VNN-COMP 输入重新建立环境，并连续运行两次。两次均：

- 退出码为 0，manifest `status=ok`；
- 与冻结的 `artifacts/rvir/rvir-cpu-correctness-v2-20260803/resnet_semantics.json`
  对照 12 个关键字段全部相等；
- 8 个非空 tensor 的逐字节 SHA256 在两次运行之间全部相等；
- external upper 仍为 `None`，符合 lower-only 请求合同。

因此，原审计报告中“当时不可现场重跑”的事实保持不改；其环境缺口已经由本后续证据关闭。

## 2. 固定来源

| 对象 | 来源/版本 | SHA256 或 commit |
|---|---|---|
| αβ-CROWN | `Verified-Intelligence/alpha-beta-CROWN` | `e5c7e17bf0488843acb77b7519f59876717a49f4` |
| auto_LiRPA submodule | upstream submodule | `5a098e8f9fb5786a428a024981d833d303921f2d` |
| VNN-COMP 2021 repo | `VNN-COMP/vnncomp2021` | `90419aadcf06cf543ce5c1706cae1059dc9fa6cf` |
| ResNet-2B ONNX | `resnet_2b.onnx` | `791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d` |
| property | `prop_0_eps_0.008.vnnlib` | `89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff` |

两个 external checkout 在运行前均无 tracked 修改。vnnlib parser 生成的 `.compiled` 文件为未跟踪
cache，不参与输入身份。

## 3. 重跑命令

先检出上表固定版本并初始化 αβ-CROWN 的 submodule。然后在 BoundFlow 的 `boundflow`
conda 环境中运行：

```bash
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate boundflow

ABCROWN_ROOT=/path/to/alpha-beta-CROWN
VNNCOMP_ROOT=/path/to/vnncomp2021
RERUN_OUT=/path/to/fresh-output

python scripts/replay_pr14_abcrown_initial_crown.py \
  --abcrown-root "$ABCROWN_ROOT" \
  --model "$VNNCOMP_ROOT/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx" \
  --vnnlib "$VNNCOMP_ROOT/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib" \
  --output-dir "$RERUN_OUT" \
  --workload-name vnncomp21-resnet2b-prop0-cpu-rvir1 \
  --device cpu \
  --warmup 0 \
  --repeats 1 \
  --backends pytorch_eager
```

第二次使用另一个空 `RERUN_OUT`。`warmup=0/repeats=1` 只用于正确性重放；host timing 不进入
任何性能结论。

## 4. 冻结摘要逐字段结果

两次运行均得到：

| 字段 | 结果 |
|---|---|
| `abcrown_commit` | `e5c7e17bf0488843acb77b7519f59876717a49f4` |
| `model_sha256` | `791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d` |
| `vnnlib_sha256` | `89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff` |
| device | `cpu` |
| intermediate bound count/source | `6` / `external_verifier` |
| intermediate bounds hash | `d51615b04dfb205afd67d2c21680ece4ca92f693157da1e32c7f8202a8e08cf1` |
| ReLU lower slope policy | `adaptive` |
| lower allclose | `true` |
| lower max abs diff | `3.0994415283203125e-06` |
| sign agreement | `9/9` |

逐字段核对脚本同时报告两次 `reference_semantics_match: True`。

## 5. 原始 tensor 稳定性

SHA256 对 tensor 的 CPU contiguous 原始字节计算，不依赖 `torch.save` 容器编码：

| tensor | shape | 两次共同 SHA256 |
|---|---:|---|
| `input_lower` | `[1,3,32,32]` | `206e69cdbd468d156e77da109280dfd502ca1548e3b3dffa175934a13a45ff3e` |
| `input_upper` | `[1,3,32,32]` | `9768a85f23ad1ecaaf79faa82ac86ea2e646916c193b93e1ca0bb5ff1cfc112a` |
| `linear_spec_c` | `[1,9,10]` | `02c50b97e31113e212418143aee96ccb502f8012970e7073d97344753dc12600` |
| `external_lower` | `[1,9]` | `e03cb7a8d8eae1925e6c79fa8d1251468dbf15da8a5dff56414cb96adc4cc570` |
| `boundflow_pytorch_eager_lower` | `[1,9]` | `ebba8a73da2df18de8292cc89a13af31dc90553c54e3c3173cb3ab8ea87e8855` |
| `boundflow_pytorch_eager_upper` | `[1,9]` | `899d27e6aa64d54f2b84323ca2aea8e8382fdd7228c50c72b1a8f92e9f718ae0` |
| `boundflow_nominal_output` | `[1,10]` | `798343a8381e9620d9b00cd5760df74a20f79f2d22f665a9da627bc6099192ac` |
| `onnx_nominal_output` | `[1,10]` | `ebe5063ff1210e5b6ec0abc211ca5e3e73da64667c7241877d8a05292b05f094` |

`external_upper=None` 在两次运行中一致。`two_run_tensor_digests_equal: True`。

## 6. 边界

- 本轮关闭的是 F5/M5 的“环境与原始数值可重跑性”缺口，不改变原审计的历史事实；
- 只验证 ResNet-2B prop0 initial plain-CROWN、external intermediate semantics 与
  BoundFlow eager CPU 等价；
- 没有 fresh CUDA、完整 VNN-COMP E2E、fused kernel 或性能主张；
- RVIR 总体状态仍是 VALIDATED-REDUCED，IR-5 仍是 VALIDATED-NO-GO。
