# 2026-08-14 — FSG3 Formal v3 父进程 Timeout 中止

## 判定

schema-v3首个正式尝试在32/36处中止。前32个fresh worker均退出0且environment admitted=true；第33个
`block-05-pos-02-B1-profile`没有进入计时失败，而是在连续GPU负载热浸后以约55°C等待冻结的post-init
`<=45°C`门禁。worker timeout为900秒，父orchestrator却在180秒终止子进程，属于父子合同不一致。

该轮不完整、没有manifest/summary/replay stdout，也不形成任何速度或baseline claim。完整目录保留为
`artifacts/fsg3-same-solver-timing/resnet2b-prop0-v3-aborted-parent-timeout-32-of-36/`；不得补跑4个位置、
不得从position 32恢复，也不得引用32个latency。

## 修正

- 父子进程总timeout固定为`1080s = 900s worker preflight + 180s execution margin`；
- timeout成为manifest中的正式preflight contract字段；
- subprocess timeout时保存partial stdout/stderr、host before/after、command、duration和
  `performance_claimed=false`到`failed_worker.json`，再fail closed；
- 新正式attempt从position 0重新生成完整36个fresh worker；
- 保持outer 50°C、post-init 45°C、poll 5s、worker preflight 900s、顺序、指标和统计门禁不变。

## 边界

这不是根据中途性能数字调参。修正只使父进程生命周期覆盖已经冻结的子进程preflight合同；v3尝试没有
完成任何统计或性能主张。下一完整attempt命名为`resnet2b-prop0-v4`，其timing/artifact schema仍为v3。
