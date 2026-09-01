# FSG4/B4-B0 Five-Fresh Runner 候选记录

日期：2026-08-18

状态：`IMPLEMENTED-B4-B0-FIVE-FRESH-RUNNER-PENDING-FORMAL-RUN`

## 实现

- typed capture新增production α feature-index/optional lookup与β location/sign/optional bias的
  raw hash lineage，并绑定两条compressed→native round-trip receipt；
- typed capture新增CUDA device、当前/default stream、priority与源alias pair事实；只准入默认
  stream、priority 0、无alias；
- 新增raw tensor payload安全序列化/反序列化，root replay不依赖生成端summary；
- 新增独立CUDA worker，以及5个fresh subprocess的raw-first artifact runner；
- replay逐个重建10个typed capture，并对5次离散结构、全部value/gradient的`2e-4`
  tolerance与sign exact门禁重算；
- 新增state/start-node/topology/shape/alpha-index/beta-location/gradient/alias/stream九类
  outer-resigned tamper probe。

## 已验证

- 单个fresh CUDA worker成功生成S/P两个capture，并以`weights_only=True`回读、typed重建；
- synthetic 5-run root summary重算通过；
- capture/schedule unit=`18 passed`；
- Mypy与Pylint门禁通过。

## 尚未关闭

本记录不等于formal artifact。runner/probe提交后才可执行5个独立进程、生成immutable artifact、
运行root replay与9/9 tamper。本状态不开放B4-B1或TIR，不包含performance claim。
