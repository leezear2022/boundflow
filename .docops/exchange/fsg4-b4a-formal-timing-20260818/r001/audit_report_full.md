# FSG4/B4-A formal timing Round 1 独立外部审计报告

- 审计身份：artifact source `46a8493557c49f327df4e70d7cdd7649227b14b9`；closure `d387a7c07ab4eacb6d06190ed4c0b4d02c88a7fb`
- 审计方式：不采信 executor summary 数字；从 worker raw、metadata、历史 Git blob 和外部 checkout 独立重算
- verdict：`approve-with-findings`

## AC1 — PASS：source、protocol 与 raw identity

- source 的 19 个 `code_revision` SHA256 全部由 `git show 46a8493:<path>` 重算一致。
- five-fresh identity 一致：manifest file SHA256=`503f304e526f0925a5cb97c4a98c8f1896abeb3c9dd53ab2021eaca3c5af8f79`，semantic manifest hash=`19c03abf9925bd10dc292be34d47f07567b2e6c46857847a21c8598e619f56c7`，source=`43d41172a7ab810621782f4a51955c677526ed88`。
- 外部 checkout HEAD 独立核对：αβ-CROWN=`e5c7e17bf0488843acb77b7519f59876717a49f4`，auto_LiRPA=`5a098e8f9fb5786a428a024981d833d303921f2d`，VNN-COMP=`90419aadcf06cf543ce5c1706cae1059dc9fa6cf`。
- model SHA256=`791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d`；property SHA256=`89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff`。
- manifest 精确覆盖 104 个非 manifest 文件，0 digest mismatch；manifest/protocol/summary canonical hash 分别为 `cf0d179c...b6a`、`0aa4ed46...485`、`46360e41...3d7`。
- artifact 与 tamper report 的 `/home/`、`/tmp/` 扫描均为 0 hit。

## AC2 — PASS：fresh sequence 与环境合同

- protocol 与 raw 逐元素重建为冻结的 6 block / 24 worker 顺序；control/profile 角色精确一致。
- 24 个 run id、24 个 worker PID 均唯一；相邻 subprocess 时间区间 overlap=`0`；returncode 全为 0，timeout=`0`。
- GPU name/UUID、runtime identity、source identity、protocol identity 的 cardinality 均为 1；external compute process=`0`，worker overlap=`0`，AC power=`24/24`。
- 24 个 outer formal preflight 均绑定 `nvidia_powerd=inactive` 与 `enforced.power.limit=55.0 W`；最终温度范围 `43..45°C`，worker 内层最终温度 `44..47°C`（内层合同上限 `50°C`）。
- 从每个 raw `environment_before/after` 重算区间增量：software thermal delta 与 power delta 各为 `651777..2052592 us`，每 worker 两者严格相等；hardware delta 全为 0。因此 coupled=`24/24`、independent=`0/24`、admitted=`24/24`。未使用生命周期累计绝对值相等替代区间增量。

## AC3 — PASS：correctness 与 activation

- 6/6 control pair 的 18 个 discrete solver/post/queue 字段 exact；final lower sign exact，最大 abs diff 按 block 为 `4.768e-7, 5.960e-7, 9.537e-7, 3.576e-7, 7.153e-7, 9.537e-7`，均过 `atol=rtol=2e-4`。
- 每 pair 独立解码并比较 19 个 float32 terminal export tensor，sign 全 exact；max abs diff 按 block 为 `4.411e-6, 3.248e-6, 1.192e-6, 3.099e-6, 4.411e-6, 3.129e-6`。
- 24/24 raw：B3 handoff/rerun=`0/1`；B4-A=`1/0` 且 lineage=`6`；provider/fallback 全 0。
- 12/12 profile：forward=`4`、bound evaluation=`10`、optimizer trace/evaluation/update=`1/10/9`。

## AC4 — PASS：性能分类独立重算

仅使用 6 个 control pair，B3/B4-A ratio：

- core wall：`[1.021616992, 1.018656514, 1.037866248, 1.016578073, 1.000195277, 1.019411468]`；geomean=`1.0189949992169265`，范围=`[1.000195277, 1.037866248]`，**未过** `1.03x`。
- query wall：`[1.006128302, 0.997851040, 1.008566064, 1.002519798, 0.996947022, 1.001597781]`；geomean=`1.0022597825638593`，worst=`0.996947022444439`，六对均过 `0.98x`。
- core GPU geomean=`1.0189919319887064`；query GPU geomean=`1.0022597197712242`。
- peak allocated/reserved 的 12 个 pair ratio 均为 `1.0`。

冻结规则的唯一结果是 `performance_candidate_admitted=false` 与 `validated-no-go-b4a-performance`；没有剔除 pair、修改门禁或挑样重跑。

## AC5 — PASS：profile attribution 与 claim 边界

- 12 个 profile closure/residual 范围均为 `[0.0016968808, 0.0018394918]`，通过 `1%/3%`。
- backward mean wall：B3=`11.811955 ms`，B4-A=`1.864271 ms`，局部 ratio=`6.3359635x`；同时 optimizer=`0.9816822x`、KFSB=`0.9865502x`、atomic commit=`0.9892278x`，存在反向波动。
- 该局部 profile 收益未升级为 whole-core claim；memory ratio=`1.0` 未形成显存 claim。
- `kernel_launch_delta=DEFERRED-TO-B4-A-KERNEL-DELTA`，`performance_claimed=false`。

## AC6 — PASS：replay 与 outer-resigned tamper

- 独立执行 root replay，stdout 与 `replay_stdout.txt` 逐字一致，summary hash=`46360e41e87917f2f5f801733fe6f13f10591cb63eff9b2675db37320a0bc3d7`。
- 独立重跑 14 类 tamper，全部修改 payload、重签 manifest file digest 与 manifest hash 后仍被拒绝。
- `environment-counter-delta` 由 `formal worker environment projection differs` 拒绝；`power-policy` 由 strict preflight payload 语义拒绝，均不是只靠外层 digest。

## AC7 — PASS（有小 finding）：regression、DocOps 与路线边界

- fixed related 11 files：`73 passed`。
- full：`1356 passed, 3 skipped, 6 warnings`。
- 11 个触达 Python 文件 Black check 通过；Pylint=`10.00/10`；5 个产品/runner 脚本 Mypy 通过；11 个 Python 路径的 scoped `git diff --check` 通过。
- `dol exchange validate`：`ok=true, errors=[]`；`dol lint --soft`：`ok=true`。
- closure 之后到审计开始前只有 DocOps/exchange/handoff 变更，无 B4-B/TIR 实现；路线文档仍明确 B4-B 关闭。

## Findings

### blocker

- 无。

### major

- 无。

### minor

- **F1 — static validation scope 表述不够精确。** `gemini_doc/change_2026-08-18_fsg4_b4a_formal_timing_internal_closure.md` 第 7 节写 `git diff --check: PASS`，但 `git diff --check adc175b d387a7c` 对整个交付会报告 raw stdout 的上游尾随空格及 prereg 文档 EOF 空行；只有 11 个触达 Python 路径的 scoped diff-check 为 PASS。建议后续 closure 精确写出 scope，并将 hash-bound raw 日志明确列为排除项，不要修改 v5 artifact。

### info

- **F2 — Mypy scope 应显式写为 5 个产品/runner 脚本。** 对这 5 个脚本使用 `--explicit-package-bases` 为 PASS；若把 6 个触达测试也放入同一次 Mypy，会出现 24 个测试 typing diagnostics。该项不影响 pytest、raw replay、artifact 或 NO-GO 分类，建议后续验证记录列出精确文件与参数。

## 最终判定

`approve-with-findings`。

同意 B4-A 关闭为 `VALIDATED-NO-GO-B4-A-PERFORMANCE`，只保留 correctness/mechanism evidence；不同意将约 `1.9%` core 收益计入累计 performance candidate/baseline。批准后只允许依据已外审 B4-0 的 `67.72%` opportunity 与 B4 总路线，**另行预注册并独立决策 B4-B**；本报告不构成自动启动 B4-B/TIR 的许可。
