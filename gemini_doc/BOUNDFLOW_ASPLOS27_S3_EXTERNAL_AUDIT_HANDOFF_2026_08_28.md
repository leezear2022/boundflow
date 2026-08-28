---
status: draft-pending-final-commit
date: 2026-08-28
type: external-audit-handoff
topic: boundflow
slug: asplos27-s3-optimizer-runtime
base-commit: ad58afb
formal-source: 1766cbcbb95466f3c4d9afda448a5e1db9bfbe36
result-commit: pending-finalization
performance-claimed: false
---

# BoundFlow ASPLOS'27 S3 external audit handoff

## 1. 审计请求

请外部审计方不采信closure/summary数字，从raw、源码与独立数学实现判定是否同意：

1. v1按原协议保持`VALIDATED-NO-GO-S3-V1`；
2. v2关闭为`VALIDATED-S3-3X-LOCAL-OPTIMIZER-V2`；
3. selected-value CUDA Graph历史机制因workspace所有权不安全被正式降级，当前VM+persistent output TIR+
   TVM persistent cuDNN workspace修复成立；
4. 只开放S4 same-solver implementation/correctness，不开放same-solver性能、complete-query、跨模型、
   总体10x或ASPLOS-ready claim。

## 2. Git与范围边界

- S3 base=`ad58afb`；
- final formal source=`1766cbcbb95466f3c4d9afda448a5e1db9bfbe36`；
- vendored TVM source=`9802f45b802225f2ea46499eec4ab7b16f64a73f`；
- result commit在executor最终落账后填入front matter；
- `.docops/exchange/gc0-1-prereg-20260826`下的异步审计文件与`docs/CIBC_for_DAC.pdf`是用户保留的
  范围外dirty文件，不得纳入S3 diff或提交判定；
- 旧S1+S2 exchange保持原样，本轮不能通过改写旧exchange来掩盖S3发现的安全问题。

## 3. Acceptance criteria

### AC1：顺序、source与失败证据

- S3预注册先于实现，v1 NO-GO先于v2稳健协议；
- failed attempt A/B/C都在成功v2之前产生且被原样保留，不进入headline；
- v2 protocol绑定parent source、12个code blob、TVM submodule revision、`cudnn_utils.cc` SHA256、模型与
  source capture；artifact无本机绝对路径泄漏；
- 成功v2必须来自空目录，不得resume partial raw。

### AC2：10/9语义与所有权

- P每个sample只建立一个prepared owner，evaluation ordinal=`0..9`、mutation=`0..8`；
- P直接调用S2 `forward/backward`，hot path不经过旧autograd Function、executor registry或
  `autograd.grad`；
- Adam、clamp、ExponentialLR及每step policy cut仍由host拥有；
- saved dense A=`0`、saved autograd history=`false`、fallback/eager/native-shadow=`0`；
- receipt强制selected graph=`0`、VM=`10`、output-copy TIR=`10`、warm DLPack=`0`。

### AC3：独立数值复核

对18个raw worker、N/D/P三方的10个step，用不import BoundFlow的脚本逐元素复核：

- lower max diff=`7.867813110351562e-06 <=2e-4`；
- compressed dα max diff=`8.288770914077759e-08 <=2e-5`；
- α before/after max diff=`4.917383193969727e-07 <=2e-5`；
- Adam exp_avg/exp_avg_sq max diff=`4.190951585769653e-08/1.057287590811029e-11`；
- lower/gradient sign exact；terminal state、ordinal、scheduler与10/9 counters一致。

### AC4：v1/v2性能统计

v1必须独立重算且保留：P/N geomean=`2.569574644743942x`、worst=`0.7595404807250521x`、P/D=
`1.9500804966538938x`，NO-GO。

v2必须核对18行inventory、每行N/D/P各30个正整数样本，再按冻结估计量重算：

- 每order三pair先取中位数；
- 六order-median P/N geomean=`3.243894370020976x`；
- worst=`3.2246091003383275x`；
- P/D geomean=`1.842216427387417x`；
- 未筛选18 pair geomean/worst=`3.2478466674781026x/3.178493381578001x`；
- 15秒cooldown只在worker之间且不进入任何latency sample。

### AC5：workspace与memory口径

- 亲读S2 runtime，确认selected Relax/cuDNN不再CUDA-graph capture，persistent output通过第5个
  `call_tir_inplace`写入；
- 亲读TVM patch，确认`ConvEntry`使用匹配的`AllocDataSpace/FreeDataSpace`，不把长期workspace放入TLS
  temporary pool；
- 现场连续fresh worker不得再出现allocator/double-free/heap corruption；
- whole-wrapper warm dynamic allocated/reserved必须披露为`13,824/0 B`，不得改写成`0/0`；审计应核对
  非零分配来自host Adam首次state/gradient，而不是compiled output或warm DLPack。

### AC6：replay、tamper与回归

- v2 replay必须从raw重算，summary hash=
  `494feff6457da88e45cf9a4906d42fac2254d6d4323d8d90732503ba6860fb6d`；
- 10类fully outer-resigned tamper全部拒绝；建议外审另造至少1类全重签攻击；
- v1 replay应在允许历史NO-GO模式下继续通过；
- targeted声明=`19 passed`；full regression=`1884 passed, 3 skipped, 6 warnings`；Black无需改动、
  mypy 12 files clean、pylint=`10.00/10`；
- `git diff --check`、`dol lint --soft`与exchange validate应通过。

### AC7：claim边界

- execution memo、claims map、current status、README、主计划、closure与change log必须一致；
- v1 NO-GO、三个failed attempt、S2 CUDA Graph安全性降级与13,824 B分配均不得遗漏；
- “3.2439x”只能描述固定ResNet2B P-anchor本地10/9 wrapper；
- artifact四个claim flag保持false；S4只开放实现/正确性与真实same-solver share归因。

## 4. 建议命令

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate boundflow
source env.sh

python scripts/run_asplos27_s3_optimizer_artifact_v2.py \
  --artifact artifacts/asplos27-s3-optimizer/resnet2b-p-anchor-v2 --replay
python scripts/probe_asplos27_s3_optimizer_v2_tamper.py \
  --artifact artifacts/asplos27-s3-optimizer/resnet2b-p-anchor-v2
pytest -q tests/test_env.py tests/test_asplos27_s2_crown_pipeline.py \
  tests/test_asplos27_s3_optimizer_pipeline.py \
  tests/test_asplos27_s3_optimizer_artifact.py \
  tests/test_asplos27_s3_optimizer_artifact_v2.py
pytest -q tests
```

外审必须另写stdlib-only脚本读取JSON/JSONL重算AC3/AC4，不能import artifact validator来代替独立复核。

## 5. 关键证据

- closure：`gemini_doc/BOUNDFLOW_ASPLOS27_S3_FORMAL_CLOSURE_2026_08_28.md`；
- prereg：`gemini_doc/BOUNDFLOW_ASPLOS27_S3_OPTIMIZER_RUNTIME_PREREG_2026_08_28.md`；
- v1/v2稳健协议：`gemini_doc/BOUNDFLOW_ASPLOS27_S3_V1_NO_GO_AND_V2_ROBUST_PREREG_2026_08_28.md`；
- change log：`gemini_doc/BOUNDFLOW_ASPLOS27_S3_CHANGE_LOG_2026_08_28.md`；
- v1/v2/failed artifacts：`artifacts/asplos27-s3-optimizer/`；
- runtime/compiler：`boundflow/runtime/asplos27_s3_optimizer_pipeline.py`、
  `boundflow/runtime/asplos27_s2_crown_pipeline.py`、
  `boundflow/backends/tvm/asplos27_s2_selected_value.py`；
- runner/replay/tamper：`scripts/run_asplos27_s3_optimizer_worker.py`、
  `scripts/run_asplos27_s3_optimizer_artifact.py`、
  `scripts/run_asplos27_s3_optimizer_artifact_v2.py`、
  `scripts/probe_asplos27_s3_optimizer_v2_tamper.py`。

## 6. 审计输出格式

请输出verdict、blocker/major/minor/info计数、AC1—AC7逐项PASS/FAIL、独立重算数字与脚本、稳定finding ID，
并明确是否同意只开放S4 same-solver implementation/correctness。
