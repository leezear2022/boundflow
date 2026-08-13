# RVIR-v4 V4-3D Live Return Assembly 修改记录

日期：2026-08-13

## 修改

- 新增truth-independent live atomic copy-out：先私有stage 12条α/β candidate path及
  `history/depths/thresholds` host packet，再一次性提交真实provider-owned tensors与host state；
- tensor write、host write或post-verify任一失败时同时恢复12条live tensors和完整host pre-image；
- 新增provider-neutral `UpdateBoundCoreReturn` assembly，覆盖lower/upper、prior bounds、C、working
  α/β/intermediate、empty batched lA、KFSB decision、clip decision及domain accounting；
- 新增真实αβ-CROWN live runner，以BoundFlow pre-state→10-evaluation/9-update optimizer→backward
  export→native KFSB→return assembly替换official `update_bounds_core`，official `update_bounds_post`、
  queue与termination保持未修改；
- fresh snapshot不再错误绑定历史artifact的浮点字节identity；每次重新计算snapshot/history/
  intermediate identity，同时继续冻结source commit、model/property/config及六层topology hash；
- snapshot仍以CPU raw-first形式保存，但live native state、input/spec/threshold、module bindings和全部
  backward/KFSB tensor按provider `c.device/c.dtype`恢复到真实CUDA设备；
- 扩展whole-core验证器，区分provider truth的3次candidate `update_bounds`与candidate replacement的
  0次调用，再对完整core/post树执行独立语义比较。
- 新增formal artifact generator/replayer与tamper probe；artifact绑定source code revision、V4-3A raw
  truth、live result、atomic receipts、official post/queue accounting与fresh GPU semantic rerun，probe覆盖
  lA/intermediate/child lower/α/decision/accounting/provider callback/atomic flag八类完全重签攻击。

## Capture-ready GPU结果

- RTX 4060 Laptop真实进程完整运行1次candidate core、official post和queue，solver继续至一次BaB
  迭代结束，`visited_domains=[6]`，status/success=`verified/true`；
- provider core/`compute_bounds`/`update_bounds` callback与fallback=`0/0/0/0`；
- 12条provider-owned路径全部原子提交，其中7条内容变化；official post接受packet并将6个child
  domains加入queue；
- final decision exact：`[[5,27],[5,32],[5,90],[5,90],[5,32],[5,90]]`；
- frozen provider truth与live candidate的完整core/post比较覆盖451 tensors、213,060个浮点sign，
  sign exact，最大绝对差`1.0669231414794922e-05 <= 2e-4`；
- `lb`、candidate child lower和六层lA均确认来自`cuda:0`，不是CPU replay；
- focused=`21 passed`，相关source mypy clean，Pylint=`10.00/10`。

## 失败诊断与修正

1. 初次监控wrapper改变KFSB stack caller identity；改为仅在candidate active window使用
   `sys.setprofile`拦截provider bound API；
2. 第二次运行将fresh浮点snapshot与旧artifact byte identity混用；拆分固定topology provenance与
   fresh semantic identity；
3. 第三次运行暴露working intermediate CPU/CUDA device不一致；assembly显式对齐provider device；
4. 首次链路成功后专用比较器又发现native KFSB/lA实际仍在CPU；将完整native执行域迁回live CUDA，
   重新运行后device与语义门禁均通过。

## 当前边界与下一动作

当前状态为`IMPLEMENTED-LIVE-RETURN / FORMAL-ARTIFACT-PENDING`。该记录证明capture-ready实现与一次
真实GPU接通，不等于V4-3D formal closure，更不等于V4-3E five-fresh或性能结论。

下一步固定source commit，生成V4-3D独立artifact、semantic replay和完全重签tamper报告；只有V4-3D
正式关闭后才准入V4-3E的五个fresh original/candidate correctness pairs。B2 timing继续关闭，
`performance_claimed=false`。
