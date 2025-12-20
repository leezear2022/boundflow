# Repository Guidelines

## Project Structure & Module Organization
- Core Python package lives in `boundflow/`: `frontends` (import/normalize), `ir` (primal/bound graphs), `planner` (global scheduling), `domains` (abstract domains), `backends` (TVM codegen), and `runtime` (execution glue).  
- Third-party sources stay in `boundflow/3rdparty` (`tvm`, `tvm-ffi`, `auto_LiRPA`); treat them as vendored submodules—avoid edits unless you are bumping upstream.  
- Development scripts sit in `scripts/`; tests are in `tests/` (smoke check in `tests/test_env.py`).

## Environment & Setup
- Use Conda env `boundflow`. From repo root run `bash scripts/install_dev.sh` (updates submodules, builds TVM/TVM-FFI, installs Auto_LiRPA editable, sets hooks).  
- If the env already exists: `conda activate boundflow && source env.sh` or `bash scripts/setup_hooks.sh` to auto-source on activation. `env.sh` exports `BOUNDFLOW_ROOT`, `TVM_HOME`, and augments `PYTHONPATH`.

## Build, Test, and Development Commands
- `bash scripts/install_dev.sh` — full bootstrap (idempotent).  
- `bash scripts/rebuild_tvm.sh` — incremental rebuild after C++/TIR changes in `boundflow/3rdparty/tvm`.  
- `pytest tests/test_env.py` — verify imports and shared libs; run after installs.  
- `pytest tests` — run the growing suite; add `-q` or `-k <pattern>` as needed.  
- After C++ changes re-run `rebuild_tvm.sh` and restart Python processes to reload shared objects.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indent, type hints everywhere; prefer dataclasses for IR containers (see `boundflow/ir/task.py`).  
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`. Keep docstrings concise and semantic.  
- Formatting/linting: run `python -m black .` and `pylint` on touched modules; keep type discipline with `mypy` when modifying planner/runtime interfaces.

## Testing Guidelines
- Use `pytest`; name files `test_*.py` and functions `test_*`.  
- Cover happy and failure paths for planner scheduling, domain transformers, and backend lowering; prefer deterministic seeds for numerical checks.  
- Add smoke cases when introducing new CLI flags or environment behaviors to avoid regressions in `env.sh` and scripts.

## Commit & Pull Request Guidelines
- No existing history—adopt Conventional Commits (e.g., `feat: add zonotope domain`, `fix: guard missing tvm lib`). One logical change per commit when possible.  
- PRs should include: short summary, linked issue (if any), commands run (`pytest`, `rebuild_tvm.sh` when relevant), and notes on performance/coverage impact. Screenshots only when changing user-facing outputs.  
- Keep third-party bumps isolated and note upstream commit SHAs when updating submodules.

## Security & Configuration Tips
- Never commit credentials or local paths; `env.sh` is the right place for machine-specific config.  
- When changing `PYTHONPATH`/`TVM_HOME` handling, verify `tests/test_env.py` and re-run activation to ensure hooks still work.

## 约定
- conda 环境是 boundflow
- 全程用中文交流
- 生成的文档放在 gemini_doc中
- 每次修改都写一个修改文档作为记录

## 关键文档索引
- TVM 后端优化备忘：`gemini_doc/tvm_backend_optimization_memo.md`
- 与大模型协作工作流模板：`gemini_doc/llm_collaboration_workflow.md`
