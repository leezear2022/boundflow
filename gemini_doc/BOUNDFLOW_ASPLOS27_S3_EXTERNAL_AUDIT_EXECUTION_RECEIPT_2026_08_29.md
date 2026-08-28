---
status: external-audit-execution-revalidated
date: 2026-08-29
type: audit-execution-receipt
topic: boundflow
slug: asplos27-s3-external-audit-execution-receipt
exchange-task: asplos27-s3-optimizer-runtime-20260828
exchange-round: 1
formal-source: 1766cbcbb95466f3c4d9afda448a5e1db9bfbe36
result-commit: 6ef12b55dfc1c9f1207adcc55694460a72e14821
revalidation-head: 3475363dff2771f128e2c5c2d6105ec0d6f7f30b
external-audit-verdict: pending
execution-authority: false
code-change-open: false
performance-claimed: false
---

# BoundFlow ASPLOS'27 S3外审执行复核回执

## 0. 结论

S3 exchange仍为`ready_for_audit/r001`，没有收到外部审计结果。本回执不充当审计，也不批准S4。

本机在2026-08-29重新执行了request中可由executor复核的全部主要命令，得到：

- S3 result commit到当前HEAD的六个关键源码/脚本/closure与formal artifact tree：零漂移；
- protocol绑定的12个code SHA256：12/12与当前文件一致；
- v2 stdlib replay：PASS，summary hash逐位一致；
- fully outer-resigned tamper：10/10 rejected；
- targeted：19 passed；
- full：1884 passed, 3 skipped, 6 warnings；
- Black：12 files unchanged；
- mypy：12 files clean；
- pylint：12个文件逐文件均10.00/10。

因此request当前可以直接交给外部模型执行。它仍必须自行重算AC3/AC4、亲读ownership源码并自建攻击；不能采信本回执
作为approve证据。

## 1. 不可变审计边界

```text
exchange task = asplos27-s3-optimizer-runtime-20260828
exchange state = ready_for_audit
exchange round = 1
approved_round = null
formal source = 1766cbcbb95466f3c4d9afda448a5e1db9bfbe36
result commit = 6ef12b55dfc1c9f1207adcc55694460a72e14821
revalidation HEAD = 3475363dff2771f128e2c5c2d6105ec0d6f7f30b
```

本回执没有修改`.docops/exchange/asplos27-s3-optimizer-runtime-20260828`中的request、delivery、state或旧round。
DocOps exchange规定旧round不可改写；若外审返回finding，只能通过immutable response与新round处理。

exchange文件SHA256：

```text
request.md   e972474ca24c54d7332eb0cbb7fc03ecffa1dc322471b22cd80aa75684b9bac5
request.json c7389074627640d6b519d3823b3b45789635979383b233f6b2ea961d040892e7
delivery.md  3e68748d405b80601afd1b8ca1d66f7d0539b1fb2a8347be2536170e4f9f11fa
delivery.json 673f7da4f6dfdbecdb1f0de36e52d7f9a6f7384a9ae78eacb3bed7d0c0a4f784
```

## 2. result commit到当前HEAD的零漂移证据

对以下范围运行：

```bash
git diff --name-status \
  6ef12b55dfc1c9f1207adcc55694460a72e14821..3475363dff2771f128e2c5c2d6105ec0d6f7f30b -- \
  artifacts/asplos27-s3-optimizer/resnet2b-p-anchor-v2 \
  boundflow/runtime/asplos27_s3_optimizer_pipeline.py \
  boundflow/runtime/asplos27_s2_crown_pipeline.py \
  boundflow/backends/tvm/asplos27_s2_selected_value.py \
  scripts/run_asplos27_s3_optimizer_artifact_v2.py \
  scripts/probe_asplos27_s3_optimizer_v2_tamper.py \
  gemini_doc/BOUNDFLOW_ASPLOS27_S3_FORMAL_CLOSURE_2026_08_28.md
```

输出为空。进一步逐文件比较result commit blob和当前工作树blob：

```text
asplos27_s3_optimizer_pipeline.py       b4f42522b4a07e4db2536828dd8a9b2387820a50 exact
asplos27_s2_crown_pipeline.py           10ec256af344d1b7cf79cf2520de08789b124427 exact
asplos27_s2_selected_value.py           7ce4c2548d9ffb54ff39d72f0fd4a7d46ca7906a exact
run_asplos27_s3_optimizer_artifact_v2.py cd282ee0c8adcd1c943636d6fb8bb4afe7dcd265 exact
probe_asplos27_s3_optimizer_v2_tamper.py d2b0d880117314ca0dedbc8c89f5ed2c9d4c5029 exact
S3 formal closure                       febe87b863270a39a03e0a7d30352b1c7a85e715 exact
```

formal artifact tree在result commit与当前HEAD均为：

```text
567ce45c864e0b2ce51a8322cd51bb4fb2d7804d
```

后续S4提交只增加或修订design文档，没有改写S3执行证据。

## 3. protocol code revision独立核对

从`protocol.json.code_revision`读取12个路径，不采信closure中的“12 files”摘要，逐文件重算SHA256：

```text
12/12 PASS
```

具体路径包括：

1. 3个S2/S3 runtime/backend文件；
2. `env.sh`与`scripts/install_dev.sh`；
3. 5个worker/artifact/tamper脚本；
4. 2个S3关键测试文件。

formal raw仍为`18`行、`20,747,422`字节；本回执没有重新生成raw，也没有改写manifest。

## 4. 当前环境

审计request原命令当前可直接工作：

```text
python = /home/lee/miniconda3/envs/boundflow/bin/python
torch = 2.12.1+cu132
cuda_available = true
cuda_device_count = 1
device0 = NVIDIA GeForce RTX 4060 Laptop GPU
```

外审应先执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate boundflow
source env.sh
```

不要根据旧会话里“conda env没有torch”的历史描述选择解释器；该描述对当前机器已经失效。

## 5. replay与tamper现场复放

### 5.1 replay

```bash
python scripts/run_asplos27_s3_optimizer_artifact_v2.py \
  --artifact artifacts/asplos27-s3-optimizer/resnet2b-p-anchor-v2 --replay
```

现场输出：

```json
{"performance_claimed":false,"status":"replay-passed","summary_hash":"494feff6457da88e45cf9a4906d42fac2254d6d4323d8d90732503ba6860fb6d","validated_s3_3x":true}
```

### 5.2 tamper

```bash
python scripts/probe_asplos27_s3_optimizer_v2_tamper.py \
  --artifact artifacts/asplos27-s3-optimizer/resnet2b-p-anchor-v2
```

结果：

```text
case_count = 10
rejected_count = 10
outer_resigned = true for all cases
performance_claimed = false
```

十类分别覆盖latency、step lower、optimizer moment、replicate index、execution counter、estimator、protocol gate、
claim flag、code revision与summary status。外审仍必须另造至少一类fully re-signed攻击。

## 6. 测试现场复跑

### 6.1 targeted

```text
19 passed, 1 warning in 16.82s
```

warning仅为PyTorch `torch.jit.script` deprecation。

### 6.2 full

```text
1884 passed, 3 skipped, 6 warnings in 726.30s
```

三个skip与closure一致：

- TVM已存在，跳过重复allow-no-tvm编译；
- 两项冻结VNN-COMP checkout在当前机器不可用。

六个warning为既有PyTorch JIT deprecation、profiler cycle提示和四项TreeSpec future warning。没有测试失败。

## 7. 静态检查及命令口径纠正

### 7.1 Black

对base/result diff中的12个Python文件运行：

```text
12 files would be left unchanged
```

Black同时提示当前Python 3.12无法执行面向Python 3.15语法的AST safety parse；format check本身通过。外审应披露该
环境提示，不应把它写成format失败或完全静默的无警告PASS。

### 7.2 mypy

第一次把`scripts/*.py`与package文件一起运行、未加package-base参数时，mypy因同一脚本被识别成顶层模块和
`scripts.*`两个名字而拒绝启动。这不是类型错误。按mypy提示使用：

```bash
python -m mypy --explicit-package-bases <12 files>
```

结果：

```text
Success: no issues found in 12 source files
```

### 7.3 pylint

一次性对12个文件做combined pylint会跨文件启用`R0801 duplicate-code`，得到`9.97/10`与exit 8；这些重复来自v1/v2
tamper与artifact测试的并列历史证据，不是本轮代码漂移。逐文件运行与closure的“10.00/10”口径一致：

```text
12/12 files: rc=0, score=10.00/10
```

因此外审不得含糊写“pylint 10.00”而不给命令口径；若采用combined invocation，应诚实报告9.97和R0801。

## 8. 外审方仍必须独立完成的工作

本回执没有完成以下事项：

1. 用stdlib-only脚本逐元素重算AC3的lower/dα/α/Adam状态误差；
2. 从18行raw独立重算AC4的六order median、geomean和worst；
3. 亲读TVM persistent cuDNN workspace与prepared output ownership；
4. 现场验证fresh worker无double-free/heap corruption；
5. 自建至少一类fully re-signed tamper；
6. 独立判断AC1—AC7和claim边界；
7. 生成正式`audit.md/audit.json`并推进exchange状态。

只有外部审计明确approve、DocOps exchange关闭且findings全部处理后，S3门禁才可关闭。届时也只开放S4-0实现/
正确性；S4 timing、same-solver speedup、complete-query、跨模型、10x和ASPLOS-ready仍保持关闭。

## 9. 交给外审模型的入口

主输入保持不变：

```text
.docops/exchange/asplos27-s3-optimizer-runtime-20260828/request.md
.docops/exchange/asplos27-s3-optimizer-runtime-20260828/r001/delivery.md
```

本回执作为执行环境和命令口径附录：

```text
gemini_doc/BOUNDFLOW_ASPLOS27_S3_EXTERNAL_AUDIT_EXECUTION_RECEIPT_2026_08_29.md
```

外审应以result commit与formal artifact为事实边界，而不是以当前HEAD新增的S4设计文档推断S3性能或正确性。
