# CIBC（AAAI’24稿）到BoundFlow ASPLOS’27候选稿的新增量草稿

date: 2026-08-27
status: draft-not-submission-ready
performance-claimed: false

## 1. CIBC既有部分

CIBC是2023年构思并提交AAAI’24但未录用的既有工作。BoundFlow候选稿不把以下内容重新声明为首次贡献：

- BoundConv/IBP lower-upper的horizontal fusion；
- center/deviation或等价interval重写；
- Conv tensor-program lowering与GPU schedule/autotuning；
- CIBC稿已出现的模型、基线、定理、实验和文字。

最终changes note需要逐段对照原CIBC PDF；本草稿只冻结边界，不替代正式差分。

## 2. BoundFlow候选新增量

1. **对象扩大**：从IBP/BoundConv扩到production αβ-CROWN/BaB exact-call region，包括active β、compressed
   α/β、split/history、incoming coefficient、optimizer 10/9 mutation和atomic publish；
2. **编译问题扩大**：从单类horizontal operator fusion扩到跨层CROWN reverse-wavefront、representation
   choice、lifetime/rematerialization、minimal-saved-state custom VJP和physical arena；
3. **系统接入扩大**：从独立算子/图runner扩到RVIR same-solver replacement，保留branch、termination、
   trajectory和verdict；
4. **证据扩大**：增加typed legality、fail-closed receipt、raw-first semantic replay、fully re-signed tamper、
   fixed-work/solved-query双scope和直接B0累计消融；S0已用explicit transaction把两个fixed-prefix workload的
   最低机制覆盖闭合到`99.632%/99.248%`，但这仍不是新系统speedup；
5. **负结果驱动的设计**：B4-C2 dense retention、MR5 bridge、MR6 guard-only和B3/B0 parity失败用于证明为何
   必须coarse region + prepared runtime，而不是被隐藏。

## 3. 仍需完成

- 对`docs/CIBC_for_DAC.pdf`逐节制作“保留/扩展/删除/新写”表；
- 检查CIBC旧图、公式、实验是否需要引用changes note或从匿名稿移除；
- 最终BoundFlow headline必须来自same-solver complete-query direct raw，不能用CIBC数字代替；
- S0归因已通过但S1性能门禁未通过；changes note不得暗示新增系统已经实现、达到预算投影或优于CIBC。
