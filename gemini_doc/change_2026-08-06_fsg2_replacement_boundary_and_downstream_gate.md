# FSG2 Replacement Boundary 与 FSG3—FSG5 依赖门禁关闭记录

## 结论

FSG2 以 **VALIDATED-REDUCED（initial-CROWN only）** 关闭；预注册的完整B2 same-solver
replacement以 **NO-GO / not admitted** 关闭。因此FSG3的B0/B1/B2 timing、FSG4的B3—B7累计消融
和FSG5系统结果均被依赖门禁阻止，未运行、也没有性能claim。

这个结论只否决当前RVIR-v3到真实αβ-CROWN production state的接入完整性，不否决BoundFlow在
算子、图IR、JIT、runtime batching/stream或arena/buffer reuse各层的潜在收益。

## 已验证的正向边界

- RVIR-v3 transport支持initial/alpha/beta phase、lower/upper/both、dense/ragged、owned tensor
  digest、显式mutable/copy-out state receipt；
- replacement backend API不接收original callable，external/provider backend identity fail closed；
- 独立Torch affine合同和mutation/shape/dtype/polarity/fallback负向路径通过；
- native initial-CROWN replacement在冻结ResNet2B property 0上：
  - lower max absolute difference=`7.152557373046875e-7`；
  - sign agreement=`9/9`；
  - replacement/original/fallback dispatch=`1/0/0`；
  - Bound/Plan/Task/Schedule/Trace五层IR hash齐全；
  - artifact summary hash=`fd6dbd43309f9e04e607b9e56a899d0a86c50f1c81372339847cf5dcd874e6c4`；
  - manifest hash=`94f06ef1b76b637a895fa8e6c166b26515514fc22dc8aa0a4ea1c56ef9bfb10a`；
  - semantic replay与source-code provenance复核均通过。

## 真实production state inventory

冻结输入：αβ-CROWN=`e5c7e17b…a49f4`、auto_LiRPA=`5a098e8f…1f2d`、VNN-COMP=
`90419aad…a6cf`、ResNet2B property 0、RTX 4060 Laptop、CUDA、seed=100、batch=64、
`max_iterations=1`、alpha/beta steps=`5/10`、isolated cold property copy。

正式v2 artifact共捕获24个真实`compute_bounds`调用：

| phase | calls | 关键状态事实 |
|---|---:|---|
| initial-CROWN | 12 | initial replacement已有独立native backend，但这里只做state inventory |
| alpha optimize | 1 | call前21个start-node keyed alpha tensors |
| beta/split | 11 | 每个call前后module显式beta tensor均为0；每call有12个`interm_bounds` tensors |
| unclassified | 0 | phase归类完整 |

另外，11个beta/split调用中`intermediate_constr`键可见，但其中tensor leaf均为0；后9个调用各有
12个`aux_reference_bounds` tensors。可见intermediate bounds不等于可独立执行的beta/split
ownership：split decision、beta materialization和optimizer mutation仍由provider内部拥有。

v2 artifact：

- `artifacts/fsg2-rvir-v3/resnet2b-production-state-inventory-v2`；
- summary hash=`37f6dbcdeadf74591ce52eaeb5d22116948ce06867a5ba3ba7ee76a23daa6544`；
- manifest hash=`e8548a25b851ac86faae3b8bb216c987c58136b9c0970ce3652dfcba559fff06`；
- semantic replay PASS，`performance_claimed=false`；
- benchmark源仓库在正式运行后保持clean。

## 为什么B2不能计时

B2定义要求“B1 + BoundFlow replacement executor，reference operators”。当前只有initial-CROWN能在
不回调original provider的情况下执行；真实alpha optimization与beta/split没有lossless executable
payload和独立BoundFlow backend。若此时计时，候选会混入original provider或漏掉solver work，违反：

- query/state/parent/order/call count exact；
- branch/node/verdict exact；
- replacement no-fallback/no-original；
- same-solver公平比较。

因此：

- B0仍是有效official control denominator；
- B1仍只是typed passthrough机制，不是BoundFlow replacement；
- B2不存在合法测量对象；
- B3—B7依赖B2，不能跳过B2把局部层收益外推成full-stack结果；
- FSG5的`>=1.20x` queue、`>=1.15x` complete-query等门槛无合法candidate分母。

## 验证

- RVIR-v3及inventory targeted：`18 passed`；
- Black通过；两个runner mypy（`--follow-imports=skip`）clean；Pylint=`10.00/10`；
- 全量（加载`env.sh`）：`1107 passed, 3 skipped, 6 warnings`，耗时`377.08s`；
- 首轮未加载`env.sh`时collection因TVM不可见产生3个环境性error，加载仓库规定环境后全部通过；
- `git diff --check`与DocOps lint在最终提交前执行。

## 后续若要重开

必须另立预注册分支，先补齐production state ownership，而不是直接进入TIR调优：

1. 冻结start-node keyed alpha的完整semantic key、batch/spec/domain axes与copy-out mutation；
2. 从solver domain/split owner显式提取decision、beta、history、intermediate/aux bounds；
3. 实现独立BoundFlow alpha-optimize与beta-split backend，禁止original callback/fallback；
4. 对24个真实call逐call复核result、state mutation、parent/depth、branch/node/verdict；
5. 只有上述门禁全过，才重新生成B2并恢复FSG3—FSG5。

