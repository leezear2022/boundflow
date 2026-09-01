# FSG4/B4-0 Kernel/Materialization Attribution 内部关闭记录

日期：2026-08-16  
状态：`INTERNALLY-VALIDATED-B4-0-OPPORTUNITY-PENDING-EXTERNAL-AUDIT`  
source：`66154e485594e8a84ad1ce04d701d8543c1a7335`  
artifact：`artifacts/fsg4-b4-kernel-attribution/resnet2b-prop0-v1/`  
性能声明：`performance_claimed=false`

> 2026-08-16更新：Round 1外审AC1—AC7全部PASS，exchange已`closed/approved`。本内部状态已由
> `EXTERNALLY-APPROVED-VALIDATED-B4-0-OPPORTUNITY`取代；外审关闭记录见
> `change_2026-08-16_fsg4_b4_0_external_audit_closure.md`。

## 1. 正式执行

从clean source `66154e4`按固定顺序运行一个fresh无profiler B3 control和一个fresh B3 profiler
worker。解释器、GPU/温度、模型/性质、三个外部仓库commit、B3正式manifest及10个code blob均进入
protocol；artifact未发现`/home/`、`/tmp/`或`file://`本机路径。

control/profile离散语义与lower sign exact，lower max abs diff=
`4.76837158203125e-07 <= atol=rtol=2e-4`。profile/control core/query wall=
`1.6409997114035897x/1.2536712453487497x`，只披露profiler扰动，不作性能数字。

## 2. Raw closure

- raw event：`270609`；真实CUDA kernel：`35367`；
- phase closure：`35367/35367`，unattributed=`0`；
- correlation parent=`33060`，显式temporal marker fallback=`2307`；
- marker：optimizer CROWN=`10`、terminal export CROWN=`1`、KFSB child CROWN=`3`；forward=`1+3`；
- raw以确定性gzip JSONL保存，manifest绑定压缩文件SHA256，worker同时绑定解压JSONL SHA256、
  canonical raw hash、行数与worker hash；summary只能从raw重建。

temporal fallback是公开归因方法，不等同correlation evidence；其事件全部保留在raw，可由外审重新按
时间包含关系复算。

## 3. Opportunity closure

从formal summary的exact phase ledger独立聚合：

| candidate | calls | kernels | CUDA kernel-sum | materialization ops | device allocation delta |
|---|---:|---:|---:|---:|---:|
| optimizer CROWN | 10 | 6657 | 24381988 ns | 2340 | 25512960 B |
| terminal export CROWN | 1 | 578 | 1117837 ns | 252 | 2851328 B |
| KFSB child CROWN | 3 | 1961 | 7118504 ns | 699 | 28928512 B |
| cumulative CROWN14 | 14 | 9196 | 32618329 ns | 3291 | 57292800 B |

`device allocation delta`是profiler事件的累计分配delta，不是unique allocation、peak memory或可直接
实现的memory saving。

B3正式raw冻结CROWN14 query share=`0.12010163988903595`、whole-core query share=
`0.17735758999613638`，两者之比=`0.6771722591159042`。因此：

- B4-A：`terminal_export.crown.00`是完整、独立、可由第10次optimizer evaluation handoff消除的重复
  CROWN call，满足“消除一个完整重复call”的准入分支；
- B4-B：CROWN14覆盖约67.72% B3 core，超过5%门槛；但若只靠该区域追回B0 query parity，仍需
  `3.989702826086512x` region speedup；
- optimizer-only无限加速仍不可追回B0，旧Amdahl否决不变。

本结论只准入B4-A terminal handoff和B4-B differentiable lower-only TIR的候选设计，不证明它们已
实现、正确或更快。

## 4. Replay、攻击与验证

- root replay：PASS；summary hash=
  `987f756db1a257877fbc1581cda85cc00f5d4e7312ab6f3219ad74d58f26bc9e`；
- manifest hash=`8720d2f9c8bb2260c1b7a8e9c328762c2a86623b36c5db3ef165825c5891c4b3`；
- 9类outer-resigned攻击独立重跑`9/9 rejected`；report hash=
  `0710e26ceed6d2623bc674978d66b6874a911ef361602bc58a927601b3a7865e`；
- targeted=`15 passed`；B3/B4相关=`54 passed`；full=`1329 passed, 3 skipped`；
- Black、Mypy、Pylint `10.00/10`、`git diff --check`：PASS。

## 5. 下一门禁

当前内部关闭，等待外部模型从raw独立复算。外审批准后只开放B4-A：在optimizer第10次evaluation中
handoff terminal lower/lA，令terminal export重复CROWN count从1降为0；先过5 fresh correctness和
`B3/B4-A core>=1.03x`/query worst pair `>=0.98x`门禁。B4-B可设计但不得与B4-A混跑；B4-C/D、
B5—B7仍关闭。
