#!/usr/bin/env bash
set -euo pipefail

# Phase 6H "one-command" runner: sweep -> report -> plot
# Output directory defaults to ./artifact_out/phase6h_<timestamp>

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${1:-}"
if [[ -z "${OUT_DIR}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="artifact_out/phase6h_${TS}"
fi

# Optional: override workloads (comma-separated, passed through to sweep).
# Default keeps kick-the-tires fast and deterministic.
WORKLOADS="${2:-${PHASE6H_WORKLOADS:-1d_relu}}"

# Keep every stage on one interpreter.  An explicit override is useful when
# the caller launches pytest with an absolute Conda Python while PATH still
# resolves `python` to the system installation.
if [[ -n "${PHASE6H_PYTHON:-}" ]]; then
  PYTHON_BIN="${PHASE6H_PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="python"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[phase6h] error: Python interpreter not found: ${PYTHON_BIN}" >&2
  echo "[phase6h] set PHASE6H_PYTHON to the boundflow environment interpreter" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
JSONL="${OUT_DIR}/phase6h_e2e.jsonl"
CSV="${OUT_DIR}/phase6h_e2e.csv"
MD="${OUT_DIR}/phase6h_e2e_summary.md"
FIGS="${OUT_DIR}/figs"
ENV_TXT="${OUT_DIR}/env.txt"
PIP_FREEZE_TXT="${OUT_DIR}/pip_freeze.txt"
CONDA_LIST_TXT="${OUT_DIR}/conda_list.txt"

echo "[phase6h] output: ${OUT_DIR}"
echo "[phase6h] workloads: ${WORKLOADS}"

{
  echo "date: $(date -Iseconds)"
  echo "pwd: $(pwd)"
  echo "git_sha: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python: ${PYTHON_BIN}"
  echo "python_version: $("${PYTHON_BIN}" -V 2>&1 || true)"
  echo "uname: $(uname -a || true)"
  echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-}"
  echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-}"
  echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS:-}"
  echo "NUMEXPR_NUM_THREADS: ${NUMEXPR_NUM_THREADS:-}"
} > "${ENV_TXT}"

"${PYTHON_BIN}" -c "import torch; print('torch_version:', torch.__version__); print('torch_num_threads:', torch.get_num_threads())" >> "${ENV_TXT}" 2>/dev/null || true

"${PYTHON_BIN}" -m pip freeze > "${PIP_FREEZE_TXT}" 2>/dev/null || true
conda list > "${CONDA_LIST_TXT}" 2>/dev/null || true

"${PYTHON_BIN}" scripts/sweep_phase6h_e2e.py \
  --out-jsonl "${JSONL}" \
  --devices cpu \
  --dtypes float32 \
  --workloads "${WORKLOADS}" \
  --ps linf \
  --specs-list 16 \
  --eps-list 1.0 \
  --max-nodes-list 256 \
  --node-batch-sizes 32 \
  --oracles alpha_beta \
  --steps-list 0 \
  --lrs 0.2 \
  --timers perf_counter \
  --warmup 1 \
  --iters 3

"${PYTHON_BIN}" scripts/report_phase6h_e2e.py \
  --in-jsonl "${JSONL}" \
  --out-csv "${CSV}" \
  --out-summary-md "${MD}"

"${PYTHON_BIN}" scripts/plot_phase6h_e2e.py \
  --in-jsonl "${JSONL}" \
  --out-dir "${FIGS}" || true

echo "[phase6h] done"
echo "  JSONL: ${JSONL}"
echo "  CSV:   ${CSV}"
echo "  MD:    ${MD}"
echo "  FIGS:  ${FIGS}"
echo "  ENV:   ${ENV_TXT}"
echo "  PIP:   ${PIP_FREEZE_TXT}"
echo "  CONDA: ${CONDA_LIST_TXT}"
