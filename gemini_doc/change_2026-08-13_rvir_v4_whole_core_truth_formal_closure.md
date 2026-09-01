# RVIR-v4 V4-3A Whole-Core Truth Formal Closure

日期：2026-08-13

## 结论

V4-3A以`VALIDATED-WHOLE-CORE-TRUTH`关闭，V4-3B native backward/lA export准入。

这只关闭original provider whole-core真值捕获与语义重放，不表示BoundFlow已经替换whole
`update_bounds_core`。V4-3、B2 same-solver timing和性能claim继续关闭。

## 正式身份

- source commit：`bfdeefc197f583b47d896142889321bbe532b3b2`；
- artifact：`artifacts/rvir-v4-whole-core-truth/resnet2b-core-v1`；
- manifest file SHA256：`0e6ed721dbf796cf8923dd57e09636f05895a1a065595ea1154b170a4a0c9818`；
- manifest internal hash：`1570e572ded581ca285ed147f519dd35d3381e77d426db16e1811df00d0b7bd9`；
- truth/summary SHA256：`d0126427…d0e9` / `d1972153…cb43`；
- core/post truth hash：`f3e043eb…69ec` / `64e70a71…175e`；
- tamper report SHA256：`dafcb893e399f0dd285d81ad5f9a485a5e117520752ad49ddff035fd19978a52`。

## 捕获范围

- official ResNet2B property 0，1 core、6 domains、24 provider calls；
- 完整`UpdateBoundCoreReturn`和post packet；
- 6组working intermediate bounds、6组KFSB入口前lA；
- 3组candidate split、3次provider `update_bounds(shortcut=True)`的`[24,1]` child lower；
- final branching decision=`[[5,27],[5,32],[5,90],[5,90],[5,32],[5,90]]`；
- `n_verified/n_splits=0/6`，solver=`verified`、success=`true`、visited domains=`[6]`。

返回后的`batched_lA`为空，因此observer在KFSB消费前捕获真实lA；不能从return object倒推出这部分真值。

## Fresh semantic replay

replay不依赖冻结tensor digest相等，而是重新执行固定αβ-CROWN/auto_LiRPA/VNN-COMP commit及同一
model/property/config，然后逐树比较：

- 451 tensors，shape/dtype/device exact；
- 213,060个浮点sign exact；
- 离散结构、inventory、branch decision和solver accounting exact；
- 两次独立fresh replay最大绝对差分别为`6.198883056640625e-06`与
  `8.821487426757812e-06`，均低于`atol=rtol=2e-4`；
- 每次有49个tensor digest和2个truth hash因合法末位漂移不同，明确证明语义门禁没有偷换成固定hash。

## 篡改门禁

六类攻击全部拒绝：

1. lA numeric full resign；
2. intermediate numeric full resign；
3. KFSB candidate child-lower full resign；
4. core/branch/post decision cross resign；
5. core accounting full resign；
6. lA field deletion resign attempt。

前五类均成功重签tensor、truth、summary、artifact files与outer manifest，static replay可通过，最终由fresh
provider semantic parity拒绝；字段删除在typed inventory层提前拒绝。

## 验证

- targeted V4-3A/V4-2 regression：`12 passed`；
- full：`1180 passed, 3 skipped`；
- mypy：4个相关source clean；
- Pylint：相关source/tests=`10.00/10`；
- Black、`git diff --check`、DocOps validate/lint均通过。

## 边界与下一动作

V4-3A artifact仍是original truth，不是candidate replacement。它明确暴露当前仍由provider承担的三次
KFSB child-bound调用。下一动作只允许V4-3B：由BoundFlow native backward导出六层lA和六层
intermediate bounds，逐层通过`2e-4`、sign与schema门禁；禁止调用provider `compute_bounds`补齐，也禁止
启动B2计时、TIR/JIT/fusion或其他性能变量。
