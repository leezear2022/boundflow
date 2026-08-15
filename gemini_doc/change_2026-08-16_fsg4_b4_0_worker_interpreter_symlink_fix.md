# FSG4/B4-0 Worker Interpreter Symlink 修复记录

日期：2026-08-16  
状态：`FIXED-VALIDATED-PENDING-CLEAN-SOURCE-FORMAL-ARTIFACT`

## 问题

B4-0正式`generate`连续两次在control worker导入阶段失败，错误为`ModuleNotFoundError: torch`；两次
都发生在求解前，部分目录仅含protocol和错误日志，没有raw worker或性能数据，并已移入系统回收站。

直接运行同一`.venv/bin/python ... worker`成功。进一步复现确认：runner在generate入口对
`abcrown_python`调用`Path.resolve()`，把虚拟环境解释器符号链接解析成uv基础解释器，因而丢失
`.venv/lib/python3.11/site-packages`。

## 修复

- 解释器路径只转为绝对路径，不解析symlink；
- 在创建artifact目录前，以与worker相同的`PYTHONNOUSERSITE/PYTHONPATH/cwd`启动独立子进程并执行
  `import boundflow, torch`；失败时明确fail closed，不再留下protocol部分产物；
- 新增虚拟环境symlink保留单测，B4 targeted增至`12 passed`。

## 边界

该修复只恢复正确worker解释器选择，不改变B3求解语义、计时协议或B4 claim。正式artifact仍须从
新clean source重新从position 0生成；此前两次失败不进入任何测量集合。
