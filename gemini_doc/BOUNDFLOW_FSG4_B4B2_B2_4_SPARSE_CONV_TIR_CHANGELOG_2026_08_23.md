---
status: validated-pending-external-audit
updated: 2026-08-23T13:57:26Z
type: changelog
topic: boundflow
slug: fsg4-b4b2-b2-4-sparse-conv-tir
stage: s01
---

# FSG4/B4-B2 B2-4 P-anchor Sparse-source Conv TIR

## 1. Internal Verdict

状态=
`VALIDATED-B4-B2-B2-4-SPARSE-CONV-P0-AND-BOUNDED-LEDGER-CORRECTNESS-PENDING-EXTERNAL-AUDIT`。

本轮完整实现并验证了：

1. P-anchor compressed alpha=`[6,86]`直接进入TIR；
2. empty beta=`[6,0]`保持ABI absent，不构造dense/zero beta；
3. P0 sparse-source Conv forward/backward correctness；
4. 12个预登记schedule candidate全部生成不同scheduled TIR、成功编译并对oracle正确；
5. candidate ledger在任何timing raw前冻结，当前无winner、无performance claim。

本轮没有做formal timing，不允许根据编译或测试墙钟选择winner，也不开放B2-5/B4-B3。

## 2. Production Mapping Diagnosis

冻结raw的P-anchor映射为：

- production alpha=`[2,1,6,86]`，合同选择direction/spec=`[0,0]`，得到compressed alpha=
  `[6,86]`；
- 三个`feature_index/{0,1,2}`各86项，对应channel/height/width；
- 86个三维坐标全部唯一且均在`[16,8,8]`范围内；
- B4-B1 native alpha gradient在这516个domain-coordinate拥有元素之外严格为零；
- production beta=`[6,0]`，location/sign=`[6,0]`，`beta_active=false`，reference beta
  gradient=`None`。

因此B2-4不需要也不允许构造`[6,16,8,8]`dense alpha作为global workspace。mapping坐标属于
PlanTemplate常量并进入template/cache hash；动态输入只包含compressed alpha。

## 3. Compiler and Runtime Changes

- 新增 sparse Conv Template/Instance/Schedule/Module/Projection/Launch/Ledger receipts；
- forward在TIR内部按常量坐标inline读取compressed alpha，完成slope/intercept、ConvTranspose
  contraction与bias reduction；
- backward直接按同一坐标计算`[6,86]`compressed alpha gradient，同时返回
  `[6,1,16,8,8]`incoming-A gradient；
- projection receipt绑定reference native gradient、reference/candidate compressed gradient、
  projected native gradient，并强制516 owned、coordinate exact、数值/sign通过、unowned zero、
  beta absent；
- DLPack/current stream exact，forward/backward exact `1/1`，fallback/eager真实计数并fail closed；
- scheduled TIR结构遍历`Block.alloc_buffers`，只准入：
  `adjoint_conv=[6,1,16,8,8]`与`output_bias_delta=[6,1]`；
- B2-3审计info #2已关闭：dense Conv新增独立shape-mismatch拒绝测试；
- B2-3审计info #1明确延期至B2-5：formal replay必须独立重编译并比对TIR/device-source hash。

## 4. Frozen Candidate Ledger

候选不是48项笛卡尔积。冻结12项balanced subset，在不超过12项的硬门禁内覆盖所有允许取值：

| ord | thread | channel tile | spatial tile | reduction unroll | schedule hash | module receipt |
|---:|---:|---:|---:|---:|---|---|
| 0 | 128 | 16 | 1 | 1 | `a4937031…55f1` | `44800f32…fce4` |
| 1 | 256 | 16 | 1 | 1 | `cd16cc8e…4120` | `3835e11b…ea83` |
| 2 | 128 | 8 | 1 | 1 | `d149828d…4360` | `ca9ecea0…6508` |
| 3 | 256 | 8 | 1 | 1 | `b98bf1af…a4fe` | `3ce2a67b…dc12` |
| 4 | 128 | 4 | 1 | 1 | `8c430a24…e983` | `593f870e…29b1` |
| 5 | 256 | 4 | 1 | 1 | `6ab7c314…1646` | `4be40a85…6ffc` |
| 6 | 128 | 16 | 2 | 1 | `118539bb…f723` | `ab5e3ef4…b398` |
| 7 | 256 | 16 | 2 | 1 | `2a1f5c5d…7af9` | `5f8ac925…56e3` |
| 8 | 128 | 8 | 2 | 1 | `8a87d387…1260` | `965d521e…aabd` |
| 9 | 256 | 8 | 2 | 1 | `02b40b64…5d42` | `ee484d41…0190` |
| 10 | 128 | 16 | 1 | 3 | `79951f61…d13c` | `b5839829…b5a2` |
| 11 | 256 | 16 | 1 | 3 | `2b2d6518…0e28` | `24995e01…8d03` |

完整ledger hash=`1660edca9f23201b14edfe8ce06947ec16f52b5b311ddb47174ea1955e8d07c6`。
12个schedule hash、scheduled TIR script与module receipt hash均唯一。ledger receipt硬校验：

- candidate count=`12`且ordinal=`0..11`；
- knobs只能来自预注册集合；
- `mapping_inline=true`；
- `generated_before_timing=true`；
- `timing_raw_present=false`；
- `winner_selected=false`；
- `performance_claimed=false`。

## 5. Correctness Evidence

### P0 five raw

- run/metrics/elements=`5/20/64,050`；
- max diff=`2.384185791015625e-06`；
- allclose/sign exact=`true/true`；
- cache=`miss,hit,hit,hit,hit`；
- module receipt=`44800f32e23693881cc7515cc8a4048eb005e1893e9a1d97fa4a7cc30851fce4`；
- 每run launch=`1/1`、fallback/eager=`0/0`、DLPack=`19/19`、beta absent。

### Twelve-candidate confirmation

- candidates/metrics/elements=`12/48/153,720`；
- 12/12 compile、execute、allclose、sign exact；
- 全候选max diff=`2.384185791015625e-06`；
- 每候选使用fresh module cache，cache event=`miss`，没有把一个module冒充12个schedule；
- 12个module receipt与schedule hash一一对应。

### Combined

- total metrics/elements=`68/217,770`；
- template=`c51b77cbdf28551cb8b97252d82a5abdda76851c5fb49e0a54547ac898f14075`；
- ledger=`1660edca9f23201b14edfe8ce06947ec16f52b5b311ddb47174ea1955e8d07c6`；
- observed workspace exact；compressed alpha=`[6,86]`；compressed beta=`[6,0]` absent。

## 6. Test and Static Validation

- B2-0 through B2-4 targeted=`51 passed in 180.23s`；
- B4-B related=`105 passed in 186.90s`；
- full=`1465 passed, 3 skipped, 6 warnings in 643.22s`；
- 3 skip均为既有环境边界：allow-no-TVM重复编译去重、两项frozen VNN-COMP checkout缺失；
- `test_env=3 passed`；
- Black check clean；Mypy 4 source clean；Pylint=`10.00/10`；diff check通过；
- TVM incremental rebuild=`ninja: no work to do`；新代码为Python TE/TIR，vendored tree未改。

## 7. Claim Boundary and Next Step

允许claim：P-anchor sparse-source Conv P0与12项bounded candidate correctness/compile ledger已验证。

禁止claim：

- 不主张任何candidate更快；
- 不主张P-anchor region speedup、memory saving或winner；
- 不主张whole-core/query/B0 parity；
- 不主张B4-B2 GO、B4-B3开放或ASPLOS-ready。

下一唯一动作=B2-4最终外审。只有外审APPROVE且0 blocker/major，才允许按照已冻结ledger进入
B2-5 formal independent-process artifact与预注册AB/BA timing；不得追加第13个candidate。

## 8. Files

- `boundflow/ir/differentiable_lower_sparse_conv_tir.py`
- `boundflow/backends/tvm/differentiable_lower_sparse_conv.py`
- `boundflow/runtime/fsg4_b4b2_sparse_conv_tir.py`
- `scripts/run_fsg4_b4b2_sparse_conv_tir_correctness.py`
- `tests/test_fsg4_b4b2_sparse_conv_tir.py`
- `tests/test_fsg4_b4b2_dense_conv_tir.py`
