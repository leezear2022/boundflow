# BoundFlow ASPLOS'27 S3 执行方影子预审

status: executor-shadow-preaudit-findings-open
date: 2026-08-29
external-audit-verdict: pending
execution-authority: false
code-change-open: false
performance-claimed: false

## 0. 结论

S3 的数值等价、10/9 optimizer 事务、性能统计、workspace 修复和当前回归均可由执行方重新复核；另一次
18-worker 临时 fresh run 也完成且没有重现 allocator 崩溃。但是，本预审发现一个必须交给外部审计裁决的
证据边界：当前所谓“10/10 fully outer-resigned tamper rejected”只覆盖“修改 raw 或派生字段、重签外层
manifest、但不同时修复全部派生语义”的攻击。若攻击者同时修改 raw、从修改后的 raw 重算 summary，再重签
manifest，replay 会接受这份内部自洽的 artifact。

这不是数值实现被证伪，也不是 replay 的普通一致性 bug。它说明离线、自签名 artifact 可以证明
`raw → summary` 的确定性闭包，却不能独自证明 raw 确实来自声明的那次 GPU 物理执行。现阶段因此不能由执行方
自行给出 approve；稳定 finding `S3-SHADOW-F1` 以 **candidate-major / open** 交给下一轮外审。

S3 exchange 保持 `ready_for_audit/r001`，旧 request、delivery、artifact 和 claims 均未修改；S4 code、formal、
timing 继续关闭。

## 1. 审计身份与边界

本文是 executor 侧 shadow pre-audit，不是独立外部审计，也不改变 DocOps exchange 状态。它的用途是：

1. 在外审前主动重算 AC1—AC7 的关键事实；
2. 把执行方发现的反例连同复现逻辑交给外审，而不是只交付有利证据；
3. 区分“artifact 内部一致性”“真实执行的新鲜性”和“外部可信锚”三类不同保证；
4. 避免将 `performance_claimed=false` 误解为无需审计性能数字——S3 的本地 `3.2439x` 仍是待审性能结果。

本文不追认外审 verdict，不开放下一阶段，不修改已经交付的 r001。

## 2. 复核对象

- exchange：`asplos27-s3-optimizer-runtime-20260828/r001`；
- formal v1：保留历史 NO-GO；
- formal v2：固定 ResNet2B P-anchor、10 evaluation / 9 mutation local optimizer wrapper；
- formal source：`1766cbc`；formal result：`6ef12b5`；
- 当前复核 HEAD：`5568039`，formal 之后仅文档/DocOps变化，protocol 绑定的12个代码文件 hash 为
  `12/12` exact；
- 对照：N=native、D=旧 D2B wrapper、P=S3 prepared local optimizer wrapper。

## 3. AC1—AC7 影子判定

| AC | executor shadow 判定 | 证据摘要 |
|---|---|---|
| AC1 顺序/source/失败证据 | PASS | 预注册→v1 NO-GO/v2协议→A/B/C失败→v2 formal 的commit祖先与时间顺序成立；A/B/C日志保留；v2拒绝非空目标且逐worker fresh subprocess，无resume。 |
| AC2 10/9语义与所有权 | PASS | 18/18 raw均为evaluation=10、optimizer/scheduler mutation=9、custom forward/backward=10；fallback/eager/native shadow=0；P路径直接调用candidate forward/backward，Adam/LR/clamp仍由host拥有。 |
| AC3 独立数值复核 | PASS | stdlib-only解析raw并逐元素计算；lower、gradient、α before/after、Adam状态均在冻结容差内，lower/gradient三对比较sign mismatch均为0。 |
| AC4 v1/v2性能统计 | PASS | v1独立重算仍为NO-GO；v2六order median、geomean、worst和P/D均与request逐位一致。 |
| AC5 workspace/memory | PASS（真实性依赖外审fresh） | TVM workspace改为ConvEntry持有并用匹配Alloc/FreeDataSpace释放；formal披露warm dynamic allocated/reserved=`13,824/0 B`；另一次18-worker fresh shadow运行完成且无allocator崩溃。 |
| AC6 replay/tamper/regression | PARTIAL / candidate-major open | formal replay、既有10/10 probe、targeted/full/static检查通过；但完整重算派生summary并重签manifest的两类攻击可被replay接受，见F1。v1历史NO-GO可由Python API replay，但CLI缺显式开关，见F2。 |
| AC7 claim边界 | PASS | claims仍为PENDING-EXTERNAL-AUDIT，仅限本地固定P-anchor trajectory；v1 NO-GO、13,824/0 B、无same-solver/query/cross-model/10x/ASPLOS-ready均保留。 |

## 4. AC3 独立数值重算

影子脚本只使用Python标准库读取18行JSONL，没有import BoundFlow artifact validator。第一版脚本因把generator
直接传给需要`len()`的geomean helper而失败；修正为先物化list后重跑，以下均为修正后的结果。

### 4.1 结构

- raw rows：18；
- latency integers：1,620；
- 每行N/D/P各30个样本；
- 每个路径10个evaluation；
- evaluation ordinal严格为`0..9`；
- `update_after == (step < 9)`；
- optimizer step严格为`min(step + 1, 9)`。

### 4.2 数值

三条pair direction全部纳入时的最大绝对差：

| 字段 | max abs diff |
|---|---:|
| lower | `7.867813110351562e-06` |
| gradient | `9.96515154838562e-08` |
| α before/after | `4.917383193969727e-07` |
| Adam exp_avg | `4.190951585769653e-08` |
| Adam exp_avg_sq | `1.057287590811029e-11` |

request/validator 披露的 gradient 最大值`8.288770914077759e-08`是以N为reference比较N/D和N/P；上表另把
P/D也直接计算在内，所以可能略大。两者口径不同，不构成矛盾。lower与gradient在P/N、P/D、N/D三组的
sign mismatch均为0。

## 5. AC4 独立性能重算

### 5.1 v2

六个order的P/N median ratio：

| order | ratio |
|---|---:|
| NDP | `3.3058681212769954x` |
| NPD | `3.231797018776845x` |
| DNP | `3.241659191800468x` |
| DPN | `3.2246091003383275x` |
| PND | `3.227018835336221x` |
| PDN | `3.2331413635374973x` |

因此：

- 六order-median P/N geomean=`3.243894370020976x`；
- worst order median=`3.2246091003383275x`；
- 六order-median P/D geomean=`1.842216427387417x`；
- 18个raw pair的P/N geomean/worst=`3.2478466674781026x / 3.178493381578001x`；
- 18个raw pair的P/D geomean/worst=`1.8392006517091517x / 1.8101876233314462x`。

headline严格使用预注册的六order median口径，没有用raw pair更有利的geomean替换。

### 5.2 v1

六个P/N pair ratio为：

`3.3462050, 3.3711660, 3.3851658, 2.9964639, 0.7595405, 3.3120343`。

独立重算：

- P/N geomean=`2.569574644743942x`；
- worst=`0.7595404807250521x`；
- P/D geomean=`1.9500804966538938x`。

因此v1必须保留`VALIDATED-NO-GO-S3-V1`，不能用v2结果覆盖。

## 6. AC2/AC5 机制与fresh shadow

### 6.1 18/18 raw receipt

每行均满足：

- evaluation / optimizer mutation / scheduler mutation=`10/9/9`；
- custom forward / backward=`10/10`；
- forward graph replay=`10`；selected value graph replay=`0`；
- selected Relax VM invocation / output copy=`10/10`；
- host policy cut=`10`；
- fallback / eager candidate / native shadow=`0/0/0`；
- saved dense A=`0`，saved autograd history=`false`；
- `performance_claimed=false`。

18行的whole-wrapper memory均为：

- allocated before=`20,391,936 B`；
- reserved before=`27,262,976 B`；
- candidate peak dynamic allocated/reserved=`13,824/0 B`。

### 6.2 ownership源码事实

- P路径直接调用prepared candidate的forward/backward；
- host创建并拥有Adam、ExponentialLR和clamp；
- selected Relax value graph在当前Torch stream上调用VM，并非被错误捕获进CUDA Graph；
- outer forward graph仍可用于其他稳定操作；
- TVM cuDNN workspace由`ConvEntry`持久拥有，分配与清理由同一CUDA device API配对。

### 6.3 额外fresh shadow

执行方在临时新目录重新生成18个fresh subprocess worker，使用当前HEAD但与formal完全相同的protocol code
blob和输入digest。结果为：

- P/N geomean=`3.2335638748978215x`；
- P/N worst order median=`3.197802647632537x`；
- P/D geomean=`1.8372285048848938x`；
- dynamic allocated/reserved=`13,824/0 B`；
- raw rows/bytes=`18 / 20,747,373`；
- 既有probe tamper=`10/10 rejected`；
- 未出现allocator、double-free或heap corruption。

该运行只说明当前机器上可再次执行成功，不是独立外审，也不能替代formal source identity；不升级claim。

## 7. Findings

### S3-SHADOW-F1 — candidate-major / open

**标题：离线自签名artifact不能证明物理执行真实性；现有10/10 probe的“fully”措辞过宽。**

执行方另造两份攻击artifact。两者均从formal artifact复制，修改raw，使用同一个validator逻辑从修改后的raw
重算所有派生summary，再重写manifest中的文件digest和canonical summary hash：

1. 把全部raw的`candidate_peak_dynamic_allocated`从`13,824`改为`0`；replay接受，并输出新的
   allocated headline=`0`；
2. 把第一行P latency减半，同时更新该行median；replay接受，并输出新的P/N headline=
   `3.2469611525481183x`。

两个artifact都“内部自洽”。因此replay接受在其当前威胁模型内是合理行为：仓库内没有独立密钥、可信时间戳、
进程/驱动attestation或外部append-only measurement anchor，validator无法区分“真实新运行”和“完整重写后自签名”。

影响：

- 既有10/10结果仍证明“不一致修改即使重签外层manifest也会被语义重算拒绝”；
- 它不能被表述成“攻击者可控制全部artifact内容时仍无法伪造一次物理执行”；
- formal数值没有因此自动失效，但其真实性必须依靠source freeze、raw-first执行纪律、失败日志、fresh rerun和
  外部审计的独立执行共同建立；
- 外审必须明确裁决该finding是major、minor还是已由独立fresh重跑充分缓解。

建议处置，不在本轮擅自实施：

1. 把历史“fully outer-resigned”准确改称“outer-manifest-resigned but derived-semantics-inconsistent”；
2. 外审现场用exact source/input运行至少一个完整fresh协议，或将不能现场运行明确列为审计边界；
3. 后续artifact若要主张执行真实性，引入独立于artifact作者的measurement anchor/签名或把该类攻击稳定标为
   `OFFLINE_UNATTESTABLE`；
4. 不用硬编码headline来“修复”validator，因为那只会把可伪造值换成另一处自签名值。

### S3-SHADOW-F2 — minor / open

**标题：v1 NO-GO replay的允许模式没有CLI入口。**

直接运行v1 artifact CLI会因3x gate不同而按设计失败；Python API传
`require_validated_3x=False`可以正确replay，summary hash为
`9ffd9deb514b09a2ff8d41fdc5d65af68ab39f3316f28562adfff51272e6306f`。request要求“允许历史NO-GO模式”
但没有给出准确调用方式。外审可先使用Python API；后续新增显式`--allow-historical-no-go`会改善可复现性，但
不应在已交付r001上静默补丁。

### S3-SHADOW-F3 — info

**标题：静态检查命令有环境/调用粒度口径。**

- mypy需`--explicit-package-bases`避免`scripts`同时被视为top-level module和package；
- 12文件combined pylint会触发跨文件历史重复代码R0801并得到`9.97/10`，逐文件均为`10.00/10`；
- Black在当前Python 3.12环境不能AST-parse配置中的未来Python 3.15 target，但12文件均unchanged。

这些已在外审执行回执中披露，不是代码回归。

### S3-SHADOW-F4 — info

**标题：fresh shadow使用当前文档HEAD，不是formal source commit。**

fresh shadow运行时HEAD晚于formal source，但protocol绑定的12个代码文件hash完全一致。它可以支持“当前环境仍能
稳定执行”，不能替代external auditor对formal source commit和物理环境的独立核对。

## 8. 已复现的回归证据

- formal v2 replay：PASS，summary hash=
  `494feff6457da88e45cf9a4906d42fac2254d6d4323d8d90732503ba6860fb6d`；
- 既有tamper probe：`10/10 rejected`，但保证范围受F1限定；
- targeted：`19 passed, 1 warning`；
- full：`1884 passed, 3 skipped, 6 warnings`；
- mypy：12文件clean（使用明确package-base口径）；
- pylint：逐文件12/12为`10.00/10`；
- code/artifact从result commit到复核HEAD：相关路径零漂移；
- formal artifact tree仍精确绑定result commit。

## 9. 给外审模型的强制问题

外审不能只复述summary，应明确回答：

1. AC3/AC4的stdlib-only重算是否成立；
2. S3-SHADOW-F1是否阻塞`VALIDATED-S3-3X-LOCAL-OPTIMIZER-V2`关闭；
3. 若不阻塞，依赖的是外审现场fresh run、现有raw-first顺序证据，还是明确降低tamper claim；
4. 是否要求在approve前修改“fully outer-resigned”措辞；
5. v1历史NO-GO的Python API replay是否足够，还是必须补CLI；
6. 是否只开放S4 implementation/correctness，继续关闭S4 formal/timing/performance。

## 10. 当前唯一下一动作

把原S3 exchange request/delivery、本影子预审和外审执行回执一并交给外部模型。外审结论回来前：

- exchange保持`ready_for_audit`；
- `blk`仍为none，因为审计尚未给出正式阻塞判定；
- S4只保留设计施工文档，不实施代码；
- 不升级本地3x为same-solver、query、跨模型、10x或ASPLOS-ready claim。
