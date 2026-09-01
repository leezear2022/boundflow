# Root compute_bounds 高层事务捕获修改记录

status: diagnostic-complete-next-compiled-prior-bound-region
date: 2026-09-01
external-audit-requested: false
performance-claimed: false

## 1. 目标

prepared root owner 已把 complete-query 提升到三组诊断 `1.08599x`，但五次 root `compute_bounds`
仍主要耗在原生 deque traversal、逐节点 Python dispatch、A/bias 装配与 concretize glue，而四段 TIR
实际 GPU 工作只有约 `7.9 ms`。

下一阶段要在 `BoundedModule.compute_bounds` 边界接管完整 root transaction。实现 replacement 前先捕获
五次 optimizer evaluation 的真实公开参数、返回结构和两个闭合公式，避免根据局部 bridge 猜 ABI。

## 2. 新增诊断

`scripts/run_bab4_root_gc_worker.py` 新增 opt-in
`--capture-root-compute-transaction`：

- 只在 root bridge 的 `_active` transaction 内捕获；
- 不捕获 init-alpha、activation-BaB 或其他 `compute_bounds`；
- 记录 method、lower/upper、return-A、final-node、C shape/hash、result shape/arity；
- 验证 terminal seed：`transpose(C @ W_final, 0, 1)` 对当前 bridge incoming A；
- 验证最终 lower：`input_concrete + pipeline_bias.T + C @ b_final` 对原生返回；
- 同时捕获更窄的 `backward_general` seam，记录 start/bound node、initial A/bias、mask、
  output-constraint 与返回合同；
- 不修改任何 tensor、optimizer 或 solver state；
- 强制 `diagnostic_only=true`、`included_in_performance_claim=false`。

## 3. 事务捕获结果

真实生产运行 `/tmp/root-backward-transaction-capture-v2.json` 闭合：

- public `compute_bounds` 恰为5次；
- 最终 `backward_general` 恰为5次，均从`/49`开始；
- `C`均为`[1,3,10]`且digest一致；
- `bound_lower=true`、`bound_upper=false`、`return_A=false`；
- initial A/lb/ub、unstable index、update mask均为空；
- terminal seed `transpose(C @ W_final, 0, 1)`最大误差`0`；
- 直接下界闭式最大误差`5.960464477539063e-08`。

因此实现了`RootCrownBackwardGeneralLiveBridgeV1`，在保留public `compute_bounds`设置、α optimizer和
host solver控制的前提下，直接接管`/49`的deque traversal。单次真实smoke达到5/5 direct、0 fallback，
离散语义与既有candidate一致。

三组交替fresh诊断位于`/tmp/bab4-root-direct-three-v1`：

- query geomean=`1.1060844872724123x`，worst=`1.0941785769391963x`；
- core geomean=`1.2230018178028532x`，worst=`1.173653818904319x`；
- root geomean=`1.0317848627386363x`，worst=`0.9762386188851344x`；
- lower max diff=`1.0132789611816406e-06`，sign/discrete exact；
- peak allocated/reserved ratio=`1.0125673800194759x / 1.0153846153846153x`；
- summary hash=`86d3c4d58ae256fc1cae69050fc456f7d29ee17de624220092de36ffeace29a2`。

相对上一批独立fresh的query `1.0859868343x`，本批约再提高`1.0185x`，但跨批差值只作诊断；本批仍
`performance_claimed=false`且未达到`1.15x`研究门槛。root scope最差pair低于1，说明deque接管不是下一主收益源。

## 4. prior-bound归因

新增opt-in `--attribute-root-prior-bounds`，对正式性能口径无效。真实运行
`/tmp/root-prior-attribution-v1.json`记录110次`compute_intermediate_bounds`，其中真正动态且有梯度的5个节点为：

| 节点 | 类型 | 5次CUDA总计 | distinct lower/upper |
|---|---|---:|---:|
| `/44` | BoundAdd | 24.264416 ms | 5 / 5 |
| `/input-28` | BoundLinear | 19.471584 ms | 5 / 5 |
| `/input-20` | BoundConv | 18.214752 ms | 5 / 5 |
| `/39` | BoundAdd | 14.971264 ms | 5 / 5 |
| `/input-8` | BoundConv | 8.821120 ms | 5 / 5 |

五项合计约`85.743 ms`。输入首层`/input`为无梯度静态值，5次仅约`0.886 ms`；其余75次主要是
无梯度参数节点的小额检查。

进一步捕获25次中间`backward_general`：每个evaluation均执行5个start-node事务；前4轮稀疏规格为
Patches `132/121/86/178`和OneHotC `27`，第5轮部分收缩为`119/85/175`，前4轮lower/upper均有梯度。

## 5. 被证伪的缓存捷径

曾在本地诊断中尝试保留第一轮中间bounds并跳过后4次prior traversal：root约`228 ms → 97 ms`，query约
`526 ms → 388 ms`。但最终BaB lower从
`[-0.3619593, …, -0.4879782]`漂移到`[-0.3829744, …, -0.5108114]`，约`2e-2`，不等价。

原因不是值能否缓存，而是五个动态bounds每轮变化且其autograd graph参与α轨迹；detach会切断VJP，直接复用
原graph又会触发second-backward错误。该实验不进入代码、不形成性能claim。

## 6. 下一实现

下一刀固定为multi-start prior-bound compiled region：

1. 将5个start node及其Patches/OneHotC稀疏规格表示成ragged start/spec axis；
2. 在共享前缀上按相同BoundOp批量传播lower/upper两条coefficient lane；
3. 复用CIBC center/radius、dual-output、residual streaming和已有custom VJP；
4. 只在kernel内物化dense A，发布5组紧凑lower/upper与六α的合并VJP；
5. 第5轮规格收缩作为dynamic active-length处理，不另编译旁路；
6. 先逐轮等价，再计时。未经等价闭合，不得采用上述`388 ms`诊断数字。
