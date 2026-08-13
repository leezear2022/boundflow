# 2026-08-13 — FSG3 功耗/温控耦合遥测修正

## 结论

FSG3 formal v1不是求解器或GPU执行失败，而是旧环境门禁把本机驱动的software power-cap/thermal
镜像遥测当成独立热降频。schema v3保留全部原始信号，只在reason和累计counter的before/after值严格
耦合时排除这个别名；任何独立software thermal或hardware thermal证据仍fail closed。

本修改不产生性能结论。正式36 fresh-process baseline仍须从clean commit的position 0重新生成。

## 独立观察

- 设备：NVIDIA GeForce RTX 4060 Laptop GPU；driver 610.57.04；AC供电；
- 受控矩阵乘观察到SW power与SW thermal累计counter从`50,539`同步增长至`168,478 us`；
- 关闭`nvidia-powerd`后镜像关系仍在，故服务不是该映射的充分原因；服务已恢复为active；
- v3 B0 control pilot：两个counter均为`50,495 -> 168,411 us`，HW thermal保持0，GPU 45→46°C，
  T.Limit margin 42→41°C，环境重新计算为admitted；
- v3 block-0 smoke：B0/B1/B2 × control/profile共6个fresh worker，6/6 admitted、语义失败0、
  profile core closure全通过、runtime identity count=1。

上述pilot/smoke均位于`/tmp`，不进入正式统计。formal v1失败证据继续保留在
`artifacts/fsg3-same-solver-timing/resnet2b-prop0-v1-aborted-post-init-thermal/`，不得追认为有效性能值。

## 官方语义核对

NVIDIA `nvidia-smi`文档将T.Limit定义为距最大运行温度的margin，并分别定义SW power capping、SW
thermal slowdown与HW thermal slowdown counter。DCGM/NVML同样为这些原因分配不同位与字段：

- <https://docs.nvidia.com/deploy/nvidia-smi/index.html>
- <https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-api/dcgm-api-field-constants.html>
- <https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html>

因此schema v3不声称“NVIDIA普遍把两者定义为同一信号”；它只记录本机driver暴露的严格镜像现象，
并以最窄规则把这一特定镜像归入已允许的ordinary power limiting。

## 实现

- timing schema升级为`boundflow.fsg3-same-solver-timing/v3`，artifact schema升级为v3；
- NVIDIA raw snapshot新增SW power reason/counter、HW thermal counter、T.Limit margin与target；
- preflight使用`independent_thermal_active`，interval gate输出software thermal、software power、
  exact coupling、hardware thermal与derived independent thermal；
- counter倒退直接拒绝；严格耦合以外的thermal证据继续拒绝；
- worker PID进入preflight，replay从before/after快照和process列表重新构造完整gate；
- 每个formal worker结束后立即刷新partial raw runs/metadata，保留fail-fast证据。

## 验证边界

已完成定向单元测试、真实单worker pilot与六路block-0 smoke。正式基线、tamper artifact、全量回归和
外部审计交接分别属于后续门禁；在它们完成前仍不得声称B2优于B0或BoundFlow已经获得端到端加速。
