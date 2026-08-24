---
status: validated-no-go-r3-1-m0-python-rematerialization
updated: 2026-08-25T06:55:00+08:00
type: changelog
topic: boundflow
slug: r3-1-m0-full-region-prototype
stage: s01
---

# BoundFlow R3-1 M0 Full-Region Prototype 修改记录

## 1. 本轮实现

- 新增 production-shaped R3-1 plan：绑定 6 组 start-node keyed compressed α/β、固定 ReLU bounds、
  split/history、16 个模型参数、input box 与 linear spec；P-anchor exact=
  `alpha/%2Finput-24/%2F49 [2,1,6,86]`、empty beta=`[6,0]`；
- 静态 registry 只保存 tensor-free plan；autograd `ctx` 只保存 plan/execution key、schema 和 alpha
  ordinal，Tensor 全部通过 `save_for_backward`；启用 `set_materialize_grads(False)` 与
  `once_differentiable`；
- custom Function forward 只返回 final lower `[6,1]`；backward 从 compressed state、bounds、weights
  重算完整 CROWN evaluation，并只返回 P-anchor compressed dα；
- worker 分离 native/candidate 进程，记录 final lower/dα、version、saved-state、allocated/reserved
  peak 与 execution receipt，不记录 latency。

## 2. 当前 pilot 结果与未通过项

单个独立 native/candidate pilot 已证明语义路径可行：final lower 最大差=`2.384e-7`，compressed dα
逐位一致；candidate forward/custom backward=`1/1`、optimizer mutation=`0`、saved dense A=`0`、
alpha/beta version 不变。

但该实现目前只能标为 **M0 Python rematerialization prototype**，不能关闭 R3-1：

- candidate/native peak allocated ratio=`1.118x > 1.0x`；peak reserved ratio=`1.0x`；
- rematerialization 仍调用 Python/PyTorch CROWN，尚未 lower 成 bounded-arena compiled region；
- 因此 dynamic allocation/module/scratch physical receipt 尚不存在，不能把静态 `scratch<=2` 当成
  物理执行证明。

## 3. Claim boundary

- `production_connected=true` 只表示输入来自冻结 production snapshot，不表示 production default
  已切换；
- `performance_claimed=false`，本轮不读取 latency；
- 不允许写成 R3-1 passed、memory improvement、compiled region 或 ASPLOS-ready；
- formal five-fresh 若仍出现 allocated ratio `>1.0` 或 compiled-region=false，应关闭为
  `VALIDATED-NO-GO-R3-1-M0-PYTHON-REMATERIALIZATION`，R3-2A继续关闭。

## 4. 下一动作

clean source five-fresh 已完成：5/5 semantic/structure 通过，但 0/5 peak allocated 通过且 0/5
compiled-region 通过，最终=`VALIDATED-NO-GO-R3-1-M0-PYTHON-REMATERIALIZATION`。只允许另立
R3-1b bounded-arena compiled recurrence，不得把当前 Python prototype 接进 optimizer。正式证据见
`BOUNDFLOW_R3_1_M0_PYTHON_REMATERIALIZATION_FORMAL_NO_GO_CLOSURE_2026_08_25.md`。
