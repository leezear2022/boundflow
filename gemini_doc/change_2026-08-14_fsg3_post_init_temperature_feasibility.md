# 2026-08-14 — FSG3 Post-Init 温度门禁可达性修正

## 结论

45°C的post-init绝对温度门禁在本机当前热稳态不可达。formal v4 attempt的第一个fresh worker在模型和
CUDA context初始化后、GPU利用率0%的情况下等待完整900秒，176次采样始终为47–54°C，后半程主要在
47–49°C回摆；所有采样的独立thermal投影均为false。该worker未执行timed query并按合同退出。

因此v4为`ENVIRONMENT-ADMISSION-INFEASIBLE`，0/36正式run，无manifest、summary或performance claim。
目录保留为
`artifacts/fsg3-same-solver-timing/resnet2b-prop0-v4-aborted-post-init-45c-infeasible/`。

## Schema v4 修正

- post-init温度上限从45°C改为inclusive 50°C；outer门禁原本就是50°C；
- 50°C是观察到的CUDA-initialized idle plateau之上的最小整数边界，51°C仍fail closed；
- independent SW/HW thermal、精确power alias规则、原始counter、T.Limit margin、AC、external process、
  worker overlap和device identity规则不变；
- timing/artifact schema升级为v4，protocol identity和manifest绑定新阈值；
- 非零worker退出现在也写`failed_worker.json`，避免只有stdout/stderr而没有结构化失败元数据；
- 新正式attempt为`resnet2b-prop0-v5`，必须从position 0完整重启。

修正后的非正式B0 control feasibility pilot在fresh CUDA进程中以首条post-init sample 47°C立即准入，
真实query完成且environment admitted=true；该pilot位于`/tmp`，不进入正式统计或性能主张。

## 方法学边界

本修订不读取、比较或选择任何B0/B1/B2 latency。v4在第一个worker计时前即失败；阈值变化只来自
可达性采样。50°C并不替代thermal event门禁：任何不与SW power严格镜像的软件thermal或任何硬件
thermal reason/counter仍拒绝。NVIDIA文档对T.Limit margin与clock-event原因的定义见：

- <https://docs.nvidia.com/deploy/nvidia-smi/index.html>
- <https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-api/dcgm-api-field-constants.html>
