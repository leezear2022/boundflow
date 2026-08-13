# RVIR-v4 V4-2E / V4-2 Formal Closure 修改记录

日期：2026-08-13

## 结论

- V4-2E以`VALIDATED-ATOMIC-COPY-OUT`关闭；
- V4-2整体以`VALIDATED-OPTIMIZER-REPLACEMENT`关闭；
- 该结论只证明固定ResNet2B production core的pre-state→10/9 native optimizer→post-state事务在
  零provider callback/fallback下正确，不等于`update_bounds_core`整核已经接入真实host solver；
- `performance_claimed=false`，B2 same-solver计时仍关闭，下一阶段是V4-3 whole-core replacement。

## 正式工件

| 证据 | 结果 |
|---|---|
| source head | `fe7cdb68d20d2691e2795713bf6ec25a92efea3c` |
| artifact | `artifacts/rvir-v4-atomic-copy-out/resnet2b-core-copy-out-v1` |
| manifest SHA256 / internal hash | `b76ee57348f2311996e6b40f013b46acdf39171a3ddc12ae2be9fa0119800136` / `05a4786707bf88e031ac80f708d7f2d1e6ef6b2c17fffbce406bcc5be5f5b40a` |
| copy-out / commit / summary hash | `d93187516245ca8669e04e3e3f1f664267a6db26af8287283bdc6f3183e34194` / `8925e836a982deeaa46a0c64ee8e8149f3af8c07a11d7ce701d9c809ce86fb71` / `d250fd49ba6ba5a4f679a4398a463027fa3ff9a9761af5fa98a8f9e031876692` |
| tamper report SHA256 / internal hash | `621d5485dd4cc72c37a61c8ef51d21e1cebac69c4bf652c8cc92a63306e6ef70` / `e15c5118eab27e27629fe08a91faaac755363626155702fe57b35136202a3abb` |

runner不采信已有summary：它从冻结V4-2D source capture重新验证production capture，重新导入ONNX、
构造native scope/pre-state、执行10次evaluation与9次Adam update，再私有stage并实际commit全部12个
mutable paths。artifact同时绑定模型、source artifact、topology和执行代码revision。

## 正式结构与数值

- `1 core / 6 domains / 6 topology rows / 10 evaluations / 9 updates`；
- `12/12` mutable paths staged和committed；candidate与production post各有`7/12` path相对pre改变；
- α/β/final lower最大绝对误差分别为`1.4662742614746094e-05`、
  `3.6135315895080566e-07`、`2.6226043701171875e-06`，均小于`2e-4`且sign exact；
- callback/fallback=`0/0`，commit receipt声明`atomic_commit=true`；
- NaN terminal、stale live target均在任何live write前拒绝；第五次copy故障注入后，前四次已写路径
  全部恢复为pre-image。

## 同步重签名攻击

original semantic replay exit 0。以下6类攻击均重算其内部hash、相关文件digest及outer manifest；source
攻击还重签source manifest，recorded-output攻击还同步重签summary与replay stdout：

1. topology internal rehash；
2. initial upper-α internal rehash；
3. expected post-α internal rehash；
4. final production lower cross-resign；
5. recorded copy-out full resign；
6. recorded commit full resign。

六类均在outer provenance与direct semantic reexecution两层拒绝，不能靠“修改payload后一起改hash”
绕过门禁。

## V4-2 Formal Acceptance 审计

| # | 预注册要求 | 证据 | 判定 |
|---|---|---|---|
| 1 | source/model/topology/policy exact | frozen digests、typed policy、model/topology replay | PASS |
| 2 | 1 core、6 domains、10/9、12 receipt、7 changed | formal summary显式门禁 | PASS |
| 3 | step/final lower/post α/post β `2e-4`且sign exact | V4-2D逐step artifact + V4-2E post artifact | PASS |
| 4 | callback/fallback `0/0` | native trace与commit receipt | PASS |
| 5 | atomic commit与失败回滚负向测试 | 12-path commit、NaN/stale/fifth-copy fault tests | PASS |
| 6 | original replay与重签tamper | replay exit 0，6/6双层拒绝 | PASS |
| 7 | focused/full/static/DocOps | focused `11 passed`；full `1175 passed, 3 skipped`；Black/mypy/Pylint通过 | PASS |
| 8 | 无性能claim，B2关闭 | artifact字段与文档边界 | PASS |

## 剩余边界与下一步

formal commit为了安全写入production-shaped isolated live clones；核心API已经验证pre-image门禁、12路提交
和回滚，但尚未替换真实host中的`update_bounds_core`，也未验证branch decision、accepted/pruned domains、
parent/depth/node accounting、termination与verdict。

下一步只启动V4-3：把V4-2 executor接入whole-core replacement，保持provider core/compute_bounds回调为
0，完成至少5次fresh correctness runs。V4-3通过后才恢复FSG3的B0/B1/B2 counterbalanced timing；
B3—B7和任何BoundFlow GPU speedup claim都不会自动升级。
