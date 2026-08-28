---
status: diagnostic-complete-code-closed
date: 2026-08-28
type: diagnostic
topic: boundflow
slug: asplos27-s4-all-state-vjp-feasibility
stage: s04
execution-authority: false
code-change-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4：六路 α / active β compiled VJP 可行性与复用审计

## 0. 结论

S4不需要推倒S2/S3，也不应为六个site各写一套独立executor。现有整图TIR已经完成了最难的结构工作：

- 六个ReLU的lower-bound coefficient传播；
- 两个残差块、Conv/Gemm、bias与input concretization；
- 六条compressed α与唯一active β的forward消费；
- 两个有界coefficient arena、persistent view、zero warm DLPack；
- P-anchor的compiled custom VJP与10/9 host optimizer资格证明。

真正缺口只有一个，但它是S4的核心缺口：当前custom backward只导出P-anchor
`alpha/%2Finput-24/%2F49`的gradient。S4要把同一整图backward扩成六条dα和一条非空dβ，并把六条空β保持为
exact empty output。

推荐实现不是保存六份dense A，而是：

```text
coefficient pass #1
  → 只保存各阶段sign bitmap
effective-value pass
  → 产生六个ReLU preactivation的选定值
coefficient pass #2
  → A到达每个ReLU时立即压缩导出dα/dβ
  → 继续复用两个arena向输入传播
```

这样跨层保存的dense coefficient A仍为0；新增持久状态是sign bitmap、六组effective primal values和compressed
gradient outputs。该方案是对现有R3/S2整图编译工作的扩展，不是另起炉灶。

## 1. production六site语义清单

数据来源：冻结production capture
`artifacts/rvir-v4-production-state/resnet2b-core-capture-v2/capture.pt`、冻结topology与R3 bounded-arena trace。

| native ReLU preactivation | provider α | logical shape | compressed width | β owner | reverse coefficient结构 | exact B4-B2单site实例 |
|---|---|---:|---:|---|---|---|
| `31` | `/48` | `[100]` | 27 | `/input-28:[6,1]` active | `Gemm_14` | sparse Linear，完全匹配 |
| `28` | `/45` | `[16,8,8]` | 178 | `/44:[6,0]` | residual11入口 | 无单siteexact实例；整图已有 |
| `25` | `/input-24` | `[16,8,8]` | 86 | `/input-20:[6,0]` | residual11内部`Conv_8` | sparse Conv，完全匹配；S3 P |
| `23` | `/input-16` | `[16,8,8]` | 121 | `/39:[6,0]` | residual6入口 | 无单siteexact实例；整图已有 |
| `19` | `/input-12` | `[16,8,8]` | 132 | `/input-8:[6,0]` | residual6内部`Conv_2` | 无该shape实例；staged residual已有 |
| `17` | `/input-4` | `[8,16,16]` | 164 | `/input:[6,0]` | `Conv_0` | 无该shape实例；整图已有 |

compressed α总宽为`708`，按`2 directions × 1 spec × 6 domains`计为8,496 stored元素。lower-only实际
optimizer-active的是`[0,0]`slice，共4,248元素；另一方向由copy-out原样保留。P-anchor宽86，对应1,032 stored/
516 active元素；两种口径下均占`12.1468926554%`。该比例不是运行时间share。

active β恰为site 31的6个元素。其location为逐domain的`[17,17,31,17,17,31]`，sign与split owner来自冻结
production snapshot/history，不得由candidate重新推断。

## 2. 已有整图能力到底覆盖了什么

### 2.1 forward不是P-only

`PreparedR31B1FullLowerForwardV1`与其S2后继已经绑定全部六个layout。现有TIR执行：

- explicit ReLU coefficient：`31/28/23/17`；
- residual11 fused/staged路径内的ReLU `25`；
- residual6 fused/staged路径内的ReLU `19`；
- active β map/split map：site `31`；
- Linear16、Linear14、Conv10/8、Conv4/2、shortcut Conv5、Conv0与input concretization。

因此S4不得把“只有P gradient输出”误写成“整图只编译了P”。forward计算与state消费已经是全图；输出ABI才是
P-only。

### 2.2 当前backward已经有的物理资产

当前`PreparedR31B2CompiledCustomBackwardV1`及D1C/D2B/S2后继已有：

1. 完整coefficient-sign reverse pass；
2. 当前43,008 bytes sign bitmap：`A24/A20/A18/Ainput`；S4另需`A26/A29`共12,288 bytes，合计55,296；
3. effective preactivation：`pre17/pre23/pre25`；
4. residual11/residual6两阶段schedule，内部coefficient分别可在stage scratch中观察；
5. P-site `dα25`压缩kernel；
6. 两个bounded float32 coefficient arena；
7. 一个logical forward和一个custom backward合同；
8. warm runtime无per-op DLPack构造、无saved dense A、无native shadow/fallback。

### 2.3 仍缺的输出与kernel

| 项目 | 当前 | S4所需 |
|---|---|---|
| effective value | `pre17/pre23/pre25` | 增加或显式导出`pre19/pre28/pre31` |
| compressed dα | 仅site 25 | site `17/19/23/25/28/31`全部输出 |
| compressed dβ | 无 | site31 `[6,1]`；其余五条exact empty |
| terminal handoff | lower + P α | lower + six gradients；ordinal 9另带lA/intermediate handoff |
| optimizer binding | 单P Adam | existing production两param-group host policy |

这三个missing effective values不要求引入新抽象IR：它们是现有selected-value/effective-value TIR图的额外输出。六路
gradient emitter应由shape/layout参数化的同一语义模板生成，formal实例仍可冻结ResNet2B shape。

## 3. 为什么不能直接串六个B4-B2单site TIR

B4-B2 dense/sparse Linear/Conv证明了局部数学与DLPack ABI，但只有两个exact production anchor：site31 Linear和
site25 Conv。直接串六个单sitewrapper有四个问题：

1. site28与site23的reverse producer是residual join，不是一个Linear/Conv；
2. site19与site17的shape不等于已冻结单sitetemplate；
3. 每sitewrapper会重新materialize incoming/output adjoint并引入Python/DLPack crossing；
4. 六个局部autograd Function无法自然保证一次logical evaluation、两个arena和terminal handoff ownership。

所以B4-B2应作为数学oracle与局部codegen资产，production实现应扩展现有整图owner。只有通用gradient emitter的
公式和拒绝逻辑适合从B4-B2抽出复用。

## 4. all-state VJP的最小算法

### 4.1 pass A：冻结本evaluation的coefficient signs

运行现有完整coefficient reverse pass。除已有`A24/A20/A18/Ainput`外，若effective-value传播确实需要额外site
sign，则只允许保存`int8` bitmap；禁止保存float32 dense coefficient。

sign receipt必须绑定evaluation ordinal、全部α/β version、plan/module hash与bitmap pointer。α/β变化后旧bitmap
必须拒绝，不能跨ordinal复用。

### 4.2 pass B：一次forward effective-value传播

从input lower/upper按sign bitmap选择端点，沿现有图正向计算：

```text
input → pre17 → ReLU17 → pre19 → ReLU19
      → residual Add6 → pre23 → ReLU23
      → pre25 → ReLU25 → residual Add11 → pre28
      → Flatten/Gemm14 → pre31
```

输出六个`[domain, feature]` effective value。它们是primal selected values，不是coefficient A，允许放在独立
persistent value arena。logical元素数为：

```text
6 × (2048 + 1024 + 1024 + 1024 + 1024 + 100) = 37,464 float32
```

即149,856 bytes；这是设计上限账，不是实测memory claim。ordinal 9 terminal lA逐site与其同shape，可在本site
gradient消费后用phase-tagged slot复用；该alias必须单独证明，S4-1 correctness不能靠未经验证的覆盖来过memory gate。

### 4.3 pass C：重算coefficient并就地压缩gradient

第二次从objective反向传播coefficient。A到达每个ReLU时，在arena被下一步覆盖前立即执行：

```text
dalpha[d,k] = upstream[d] * A_relu[d, feature(k)] * effective_pre[d, feature(k)]
```

准入条件与现有P kernel一致：lower-bound direction、ambiguous ReLU、`A_relu >= 0`、合法feature mapping；另一α
direction按当前lower-only语义输出0。随后原地应用ReLU coefficient transform并继续传播。

site31 active β按已经在B4-B2 sparse Linear独立验证的语义：

```text
dbeta[d,q] = -upstream[d] * effective_pre31[d, location(d,q)] * split_sign[d,q]
```

empty β不launch计算kernel，返回S4-1A ordered ABI中的exact typed empty token。

精确的通用`[D,S,F]→[D,W]` α emitter、site31 sparse β emitter、六site插入点与ordinal-9 terminal lA phase
合同见`gemini_doc/BOUNDFLOW_ASPLOS27_S4_1C_COMPRESSED_GRADIENT_EMITTER_IMPLEMENTATION_BLUEPRINT_2026_08_28.md`。

residual11 stage1 scratch就是site25的incoming coefficient；residual6 stage1 scratch就是site19的incoming
coefficient。应在stage1与stage2之间发射compressed gradient，不再重做独立Conv。

### 4.4 结构账

S4-1允许：

- coefficient float arena：恰2个，复用现有容量；
- cross-layer saved dense A：0；
- sign bitmap：显式计数与bytes；
- effective primal arena：显式计数与bytes；
- compressed output：六dα + 一active dβ + 五empty dβ；
- Python-visible per-site tensor：0；
- warm DLPack construction：0；
- dynamic output allocation：0。

不得把effective primal value误记为dense A，但receipt必须分别披露两者，防止通过改名隐藏内存。

## 5. S4-1内部实施顺序

S3外审批准且S4转为execution-authority后，S4-1仍按以下四刀推进：

1. `S4-1A all-state ABI`：从六个layout生成ordered output slots、persistent buffers和coverage receipt；
2. `S4-1B effective values`：补齐六site selected preactivation，与独立PyTorch/f64局部oracle比较；精确arena、
   A26/A29 sign、selected Relax graph和negative门禁见S4-1B实施蓝图；
3. `S4-1C gradient emitters`：一个通用α模板实例化六site，site31另有active β；插入顺序为
   31→28→25(stage)→23→19(stage)→17；
4. `S4-1D evaluator closure`：一个logical evaluation返回lower、六dα、六dβ，five fresh通过。

1B/1C均不接optimizer、不计时；1D通过前S4-2保持关闭。

## 6. 预期风险与kill gate

1. **sign语义不足**：若现有四bitmap不足以重建六site effective values，可增加bitmap，但不得改存dense A；
2. **residual内部A不可见**：优先使用已完成的D1C/D2B stage scratch；若必须重跑native residual则NO-GO；
3. **active β公式漂移**：以B4-B2 sparse Linear和production autograd双oracle为准，任何location/sign偏差即STOP；
4. **两次coefficient pass成本过高**：S4-1只做correctness。性能问题留S4-P实测，不能在正确性阶段删状态；
5. **terminal lA需要第三次CROWN**：ordinal 9必须复用pass C最终arenas/receipt组装handoff；若需要完整第11次CROWN，
   S4-3 NO-GO；
6. **shape参数化退化为模型特判**：template允许shape实例化，但schema/capability不能出现model名或固定node id。

## 7. 当前门禁

本诊断只证明“现有资产可以形成一个明确、可证伪的S4实现方案”，不证明六路VJP已实现或会更快。S3 exchange
`asplos27-s3-optimizer-runtime-20260828`独立外审仍是唯一外部门禁；批准前S4代码与GPU执行继续关闭。
