---
status: ready-for-external-audit
date: 2026-08-28
type: external-audit-handoff
topic: boundflow
slug: asplos27-s1-s2-combined
base-commit: d7adec6
result-commit: cd1670c
performance-claimed: false
---

# BoundFlow ASPLOS’27 S1+S2 combined external audit handoff

## 1. 审计结论请求

请外部审计方独立判定是否同意：

1. 关闭`VALIDATED-S1-CIBC-CANONICAL-PIPELINE`；
2. 关闭`VALIDATED-S2-4X-CANONICAL-CROWN`；
3. 只开放S3 optimizer 10 evaluation / 9 mutation trajectory预注册；
4. 不开放same-solver、complete-query、总体10×或ASPLOS-ready claim。

不要采信closure/changelog里的数字；必须从formal raw独立重算。

## 2. Git边界

- base=`d7adec6`；
- result=`cd1670c`；
- S2 formal execution source=`d9582b552348c534dc5fb039231496e21c9f9b4c`；
- `d9582b5..cd1670c`只应包含formal artifact、closure与权威文档同步，不应修改S2执行语义；
- 仓库中GC0-1旧exchange和`docs/CIBC_for_DAC.pdf`为用户保留的范围外文件，不得纳入S1/S2 diff
  判定。

## 3. Acceptance criteria

### AC1：source/build identity

- S2 protocol逐blob绑定compiler/runtime/worker/artifact/test/env/install脚本到`d9582b5`；
- source capture、ONNX model、plan、trace、source/partitioned/lowered Relax、device sources hash一致；
- TVM build确实启用cuDNN/cuBLAS，wheel `libcudnn.so.9`可解析；缺cuDNN应fail closed；
- artifact不得泄漏本机绝对路径。

### AC2：compiler/runtime结构

- selected-value图是standard Relax数据流，不是旧serial effective TIR的包装；
- Conv0/2/4/shortcut5/8恰有5个cuDNN call sites；Conv4/8共享签名，所以4个partition functions合法；
- input/ReLU17/19/23恰有4个scheduled TIR；
- forward wavefront与selected-value chain各一次CUDA Graph replay；
- active β真实存在，two-slot arena、saved dense A=`0`、saved autograd history=`false`；
- prepare DLPack views=`29`，warm DLPack/fallback/eager/native shadow=`0`。

### AC3：三方correctness

独立从6个raw worker逐元素比较`N` native、`D`旧D2B、`P` S2：

- 每worker lower 6元素、gradient 1032元素；
- lower max diff应为`3.0994415283203125e-06 <= 2e-4`；
- gradient max diff应为`6.146728992462158e-08 <= 2e-4`；
- lower/gradient sign exact；
- receipt必须绑定同一plan/trace，篡改`active_beta`或dense-A owner应拒绝。

### AC4：六fresh性能

从`raw/workers.jsonl`独立重算，而非读取summary：

- 6行与`NDP/NPD/DNP/DPN/PND/PDN`逐一对应；
- 每行N/D/P各30个正整数host latency，median必须由raw精确重算；
- P/N geomean应为`4.24538196457207x >=4.00x`；
- P/N worst应为`3.540798856743263x >=3.50x`；
- P/D geomean应为`2.4676101727573547x >=0.90x`；
- warm dynamic allocated/reserved应为`0/0`；cold prepare单报，未混入warm headline；
- 比较仅能形成P-anchor single-evaluation same-scope claim。

### AC5：replay/tamper

- `run_asplos27_s2_crown_artifact.py --replay`必须从raw重算且summary hash=
  `694c011ae80fa4131c2fcc3112bfcd75ae1ab4e502763797662e6fb2755482e4`；
- 10类tamper全部拒绝；请至少抽查latency case确实重算median/summary、receipt case确实重签inner
  receipt、每类都重签outer manifest；
- 建议审计方再自建至少1类全重签攻击，验证不是只认冻结case名字。

### AC6：回归与静态质量

- S2专项应为`7 passed`；
- 全量声明为`1876 passed, 3 skipped`，请核对skip理由是否全为既有环境边界；
- black clean、production mypy clean、相关6文件pylint `10.00/10`；
- `git diff --check`、`dol exchange validate`、`dol lint --soft`通过。

### AC7：claim边界与文档一致性

- claims map、execution memo、current status、README、S2 plan/changelog/closure应给出同一结论与下一步；
- 历史S1“下一做S2”必须有被上方S2关闭语义取代；
- 不得出现S2已经证明optimizer、same-solver、complete-query、跨模型、总体10×或ASPLOS-ready的措辞；
- S0的10×预算仍是可证伪目标，不是实测结果。

## 4. 建议复核命令

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate boundflow
source env.sh

python scripts/run_asplos27_s2_crown_artifact.py \
  --artifact artifacts/asplos27-s2-crown-pipeline/resnet2b-p-anchor-v2 \
  --replay
pytest -q tests/test_asplos27_s2_crown_pipeline.py
pytest -q tests
python -m black --check <S2 touched Python files>
mypy boundflow/backends/tvm/asplos27_s2_selected_value.py \
  boundflow/runtime/asplos27_s2_crown_pipeline.py
pylint <6 S2 related files>
```

审计方应另写stdlib-only脚本读取JSON/JSONL重算AC3/AC4；不要import BoundFlow来重算headline。

## 5. 关键证据入口

- S1 closure：`gemini_doc/BOUNDFLOW_ASPLOS27_S1_CANONICAL_CIBC_PIPELINE_FORMAL_CLOSURE_2026_08_28.md`；
- S1 artifact：`artifacts/asplos27-s1-cibc-pipeline/resnet2b-prop0-v2`；
- S2 plan：`gemini_doc/BOUNDFLOW_ASPLOS27_S2_COARSE_CROWN_CUSTOM_VJP_PLAN_2026_08_28.md`；
- S2 closure：`gemini_doc/BOUNDFLOW_ASPLOS27_S2_FORMAL_CLOSURE_2026_08_28.md`；
- S2 artifact：`artifacts/asplos27-s2-crown-pipeline/resnet2b-p-anchor-v2`；
- S2 compiler/runtime：`boundflow/backends/tvm/asplos27_s2_selected_value.py`、
  `boundflow/runtime/asplos27_s2_crown_pipeline.py`；
- replay/tamper：`scripts/run_asplos27_s2_crown_artifact.py`、
  `scripts/probe_asplos27_s2_crown_tamper.py`。

## 6. 已知限制

- 本轮只有ResNet2B P-anchor、6 domains和一次evaluation；
- CUDA Graph依赖固定pointer/shape/schedule；dynamic α是固定owner内容更新，不是任意dynamic shape；
- warm memory数字是PyTorch allocator口径，不覆盖所有cuDNN内部不可见allocator；
- `PDN`是最差pair，仅高于3.50门槛约1.17%，请重点检查其raw，不得隐藏；
- formal不包含10/9 optimizer、RVIR exact-call或complete-query。

## 7. 审计产出格式

请输出：

- verdict：approve / changes_requested；
- blocker / major / minor / info计数；
- AC1—AC7逐项PASS/FAIL及独立证据；
- 独立重算脚本/命令与关键数字；
- findings（稳定ID F1...）；
- 是否同意只开放S3预注册。
