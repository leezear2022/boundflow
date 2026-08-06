---
status: validated-no-go
updated: 2026-08-06T07:59:16Z
type: change
topic: boundflow
slug: nrir49a-g1-gpu-attribution-nogo
stage: s01
---

# NRIR49A G1 GPU Attribution NO-GO

## 变更

- 新增只读GPU归因runner与10项合同测试；production runtime、TIR、kernel和默认chunk均未修改。
- 以clauses 2/3、31 nodes、15 sibling groups、depth 4完成5个fresh-process worker；chunk采用
  `8/16/128/32/64` Latin轮转，default 32另有paired control。
- 将代表性CUPTI范围限制为一个从真实queue捕获的child selected-CROWN调用，避免整queue profiler
  造成的26.5 GiB主机OOM；profile证据排除在timing summary之外。
- GPU parity采用离散结构exact、raw finite浮点按预注册`atol=rtol=2e-4`、完整payload hash绑定；
  alpha/beta/bounds/score等数值派生hash差异显式计数，不以bitwise hash误判ULP漂移。
- 生成9文件formal artifact和从raw worker重算summary/decision的fail-closed replay。

## 正式结果

- 服务：`boundflow-nrir49a-g1-r3.service`，exit 0，wall `30m54s`，主机内存峰值`2.1 GiB`。
- queue selected-CROWN share中位=`0.07098631834282758`；complete share中位=
  `0.070523288963519`。
- paired profiler/control ratio中位：clause 2=`0.999304435327957`，clause 3=
  `1.0067470427656482`，均通过`<=1.05`门禁。
- 60/60结构比较exact；最大absolute/relative浮点差异=
  `2.288818359375e-05/0.0001710717646052519`；数值派生hash差异总数`33877`。
- 最大allocated/reserved比例=`0.009964162844036697/0.0135301156761069`；合法domain batch上限1、
  无真实OOM，physical-memory path=`N/A`。
- CUPTI代表调用：5954 kernels、5486 runtime launches、398 sync、5364 memory events。

## 判定

`s_queue <20%`，故selected-CROWN不是GPU winner。queue `1.20x`与complete `1.15x`目标均超过相应
Amdahl无限区域加速上限，三个required-region-speedup字段均为`null`。memory admission亦失败。
G1按预注册规则以`VALIDATED-NO-GO`关闭，G2/G3 gated off；不得启动selected-CROWN TIR/JIT/融合，
下一动作是重新归因GPU whole-queue winner。`performance_claimed=false`，不形成speedup、竞品、
multi-workload、solved verdict、memory headline或ASPLOS-ready claim。

## 工件与复核

目录：

```text
artifacts/nrir49a-g1-gpu-attribution/
  resnet2b-prop0-clauses2-3-rtx4060-five-repeat-v1/
```

- summary hash：`7eefe6a716fa57874420bcda64487ad02578dae5926fc99426eaaef37d35ab50`
- manifest hash：`d0272fe431d68ba93ef17a69fe7bf9b7ef71c7ab02041680720e65fadd86c81f`
- 行数：2 queries、5 raw workers、50 normalized、0 failures。

独立replay：

```bash
conda run --no-capture-output -n boundflow python \
  scripts/run_nrir49a_g1_gpu_attribution.py replay \
  --artifact-dir artifacts/nrir49a-g1-gpu-attribution/\
resnet2b-prop0-clauses2-3-rtx4060-five-repeat-v1
```

预期核心输出：`status=replay-passed`、summary hash如上，decision为instrumentation PASS、queue
opportunity FAIL、latency feasibility FAIL、memory admission FAIL、next route=
`gpu-winner-reselection`。本轮已独立确认exit 0、stdout逐字一致、全部payload SHA256与manifest hash
重算吻合。

## 失败历史边界

- 两次前台执行和一次`nohup`被会话cgroup回收，均无worker JSON，不计入正式重复。
- retry-1整queue CUPTI触发主机OOM，无worker JSON，不计入正式重复。
- retry-2在bitwise语义hash门禁失败，无有效worker；保留diagnostic raw并据预注册浮点容差修正
  structure/numeric parity，不修改任何性能门槛。
- 正式结果只来自retry-3的5个完整fresh workers。
