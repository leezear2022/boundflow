# ASPLOS'27 S4-1A ordered buffer 外部审计交接

date: 2026-08-30  
requested-verdict: approve-or-request-changes  
requested-close: VALIDATED-S4-1A-ORDERED-BUFFER-PREPARE  
performance-claimed: false

## 1. 审计对象

- branch：`feat/rvir-v4-production-state-ownership-v1`；
- production code：`8834aa5`；unit：`29bcaa0`；
- formal source：`bce26f0d8109f69d520dfe27a04fb9c2110b34b0`；
- frozen artifact commit：`00893c5`；
- artifact：`artifacts/asplos27-s4-1a-buffer/resnet2b-prop0-v1`；
- tamper：`artifacts/asplos27-s4-1a-buffer/resnet2b-prop0-v1-tamper-report.json`；
- contract：`gemini_doc/BOUNDFLOW_ASPLOS27_S4_1A_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`。

请不要采信changelog/summary聚合数字，应从raw、binary sidecar与源码独立重算。

## 2. AC1：source与scope

核对protocol绑定的9个code blob、formal source commit、αβ-CROWN/auto_LiRPA/VNN-COMP commit及model/property
SHA256；artifact不得含本机路径。确认S4-1A只做buffer prepare，没有CROWN evaluator、TIR launch、optimizer、
mutation、terminal、timing或performance路径。

## 3. AC2：ordered physical owner

亲读production实现，确认S4-0 ticket只消费一次；资源顺序为6 α parameter→1 active β parameter→6 dα→1 dβ→
lower→upstream；5个empty β只有typed token。核对16个storage互异、candidate/source无alias、完整view key含
storage identity/pointer/nbytes、tensor pointer、shape/stride/offset/dtype/device，并在cache lookup前拒绝noncontiguous。

## 4. AC3：5个fresh正向与二进制语义

从`raw/workers.jsonl`和5个`.bin` sidecar独立重算：

- 5/5真实production exact-call；
- 每run 8组source/candidate，共40组逐字节相等；
- parameter/gradient=`4,254/4,254` elements、`17,016/17,016 B`；
- storage/view=`16/16`、candidate logical=`34,080 B`；
- empty β token/physical=`5/0`；
- S4-1A D2H=`32/85,056 B`、累计=`56/153,072 B`、D2D=`7/17,016 B`；
- close后candidate allocated delta 5/5为0。

## 5. AC4：7个fresh隔离故障

逐个核对parameter、gradient、output、TVM view、roundtrip、receipt、adoption fault各在独立新进程执行；stable
detail/reason与冻结映射一致，`__context__ is None`、allocated delta=0、fallback/retry/empty-cache=0。建议外审至少
亲启一个最早fault和一个最晚fault，并保留异常对象后复核candidate释放；最好重跑全部7个。

## 6. AC5：unit、replay与tamper

- negative registry应为77个唯一nodeid且逐项存在于测试源码，门槛为68；
- unit应为80 passed，artifact+unit应为84 passed；
- 全量应为`2050 passed, 3 skipped, 6 warnings`，3个skip均为既有环境边界；
- stdlib replay不得import BoundFlow/PyTorch/TVM/Numpy；
- 10类outer-resigned攻击应10/10拒绝，包括binary candidate、storage/view、empty β、D2H账、fault detail/cleanup、
  claim、ordinal与negative registry；
- 请自建至少一类未注册攻击，并说明是identity还是derived semantics拒绝；
- coherent full resign的E0边界仍需明确，不得把tamper probe写成硬件真实性证明。

## 7. AC6：回归与静态门禁

建议运行：

```bash
source /home/lee/miniconda3/etc/profile.d/conda.sh
conda activate boundflow
source env.sh
pytest -q tests/test_asplos27_s4_ordered_buffer_abi.py \
  tests/test_asplos27_s4_1a_buffer_artifact.py
pytest -q tests/test_asplos27_s4_mutable_state_admission.py \
  tests/test_asplos27_s4_ordered_buffer_abi.py
pytest -q tests
python scripts/replay_asplos27_s4_1a_buffer_stdlib.py \
  --artifact artifacts/asplos27-s4-1a-buffer/resnet2b-prop0-v1
python scripts/probe_asplos27_s4_1a_buffer_tamper.py \
  --artifact artifacts/asplos27-s4-1a-buffer/resnet2b-prop0-v1
```

另核对Black、scoped Mypy、Pylint 10.00、`git diff --check`与`dol lint --soft`。

## 8. AC7：判定与后继门禁

批准只允许关闭`VALIDATED-S4-1A-ORDERED-BUFFER-PREPARE`，并另行开放S4-1B0
implementation/correctness。不得升级CROWN numeric、optimizer、same-solver、memory/performance、complete-query、
10x或ASPLOS-ready claim；S4 timing保持关闭。

请输出AC1—AC7逐项PASS/FAIL、独立数字、findings分级、不可现场复核项，以及是否批准executor关闭S4-1A并只开放
S4-1B0。
