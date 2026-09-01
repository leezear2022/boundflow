# FSG4/B3 五组 Fresh Correctness 正式关闭记录

日期：2026-08-14
状态：`VALIDATED-B3-FIVE-FRESH-CORRECTNESS`

## 结论

source `75dfd8103e8e3dfe824a63e15c2222f8742e28c1`按预注册固定顺序完成10个独立fresh GPU worker。
5/5组B2/B3-C direct semantic comparison全部通过，且environment、provider/fallback、physical counter和
B3-C post-query audit均满足门禁。该结果关闭正式计时前的正确性准入，但不包含性能结论。

## 正式证据

- artifact：`artifacts/fsg4-b3-correctness-pairs/resnet2b-prop0-v1/`；
- tamper：`artifacts/fsg4-b3-correctness-pairs/resnet2b-prop0-v1-tamper-report.json`；
- source：`75dfd8103e8e3dfe824a63e15c2222f8742e28c1`；
- protocol internal hash：`241f9b7573b50b3bac11cee06937626c1141c271a78a51c9306daeee732051d6`；
- manifest internal hash：`457ab1adc8488c5353ec66294583e7a2bedf2e92fca5901a72a41e8321df1573`；
- report internal hash：`0d649200f423875db23ee23660447f7f0a8b91ce91510f254d7b3bb8f8a2827d`；
- protocol/manifest/report file SHA256：`34a3426de1260efdd97b4e50e4135f495a4e5025ac2365821ea37e98644cbd49`/
  `bf8b3ecccea992cce9dca56c963518510af8dc8d410c0d02b94513160189cb98`/
  `aab4620b147579d85edc8fb51fb4a19c8660ae5c03332b12b1a70e6c5f3d19e5`；
- tamper internal hash：`52dd43fdbb4de8411c52e31e34006e191e5d3e3cbc57727a0a0f964a0cf32798`；
- tamper file SHA256：`7f11a97ac272511289b087c77ab9a003c6766a14c91a109de450e4c0fc8bfce9`。

## 固定运行与验收结果

运行顺序没有根据中间结果调整：

| Pair | Position 0 | Position 1 | 结果 |
|---:|---|---|---|
| 0 | B2 | B3-C | PASS |
| 1 | B3-C | B2 | PASS |
| 2 | B2 | B3-C | PASS |
| 3 | B3-C | B2 | PASS |
| 4 | B2 | B3-C | PASS |

- 10/10 raw worker均由独立diagnostic subprocess生成并可独立replay；
- 5/5 pair的status/success/domain、queue、depth/history、shape/inf、decision、split与termination离散字段
  exact，lower/finite upper满足冻结`atol=rtol=2e-4`且B3-C不optimistic；
- 5/5 pair的source/protocol/runtime/GPU identity一致，environment admitted；
- provider core/compute/update/fallback在全部worker中为`0/0/0/0`；
- 每个B2 worker含4625 events，保持module/scope=`1/2`、snapshots/forward/D2H=`10/5/12`；
- 每个B3-C worker含1484 events，保持template hit=`1`、module move/snapshots/D2H=`0/0/0`、scope=
  `1`、forward=`4`、optimizer=`10/9`、KFSB=`3/3`、commit/backup/copy=`12/12/12`；
- 每个B3-C worker的headline content digest=`0`，post-query audit在计时外完成并绑定assembly、commit和
  audit hash；
- root replay不采信root report，而是从10个raw worker重放并重建5组比较。

## 篡改门禁

七类攻击在修改payload并同步重签外层hash/digest后仍全部被语义重算拒绝：

1. root report projection；
2. protocol schedule；
3. nested counter/event journal；
4. nested semantic payload；
5. nested audit receipt；
6. pair position交换；
7. raw worker删除。

结果=`7/7 rejected`。

## 验证

- root replay：PASS；
- frozen artifact与B3全部定向：`56 passed in 7.13s`；
- full：`1289 passed, 3 skipped, 6 warnings in 451.67s`；
- Pylint：`10.00/10`；
- `git diff --check`：PASS。

第一次全量收集曾因非交互shell覆盖激活钩子的TVM `PYTHONPATH`而出现3个import error；加载`env.sh`后
同一Conda解释器完成上述全量PASS。该错误不是代码或artifact回归。

## Claim 边界与下一步

本关闭只证明B3-C累计候选经过5组fresh对照仍保持正确性和物理激活。五组artifact有意保留
`timing_admitted=false`、`performance_claimed=false`，不读取或比较raw timing字段。

现在可以设置阶段门禁`b3_timing_admitted=true`，下一唯一动作是实现并验证预注册的六个B0/B2/B3
全排列、36-process正式计时runner，然后从新的clean source运行、replay和tamper。正式计时关闭前不得
声称B3 speedup，也不得启动B4 TIR、B5 JIT、B6 runtime或B7 arena。
