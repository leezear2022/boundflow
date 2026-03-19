#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly DEFAULT_TARGETS=("gemini" "claude" "codex")

declare -A PACKAGES=(
    ["gemini"]="@google/gemini-cli"
    ["claude"]="@anthropic-ai/claude-code"
    ["codex"]="@openai/codex"
)

declare -A BINARIES=(
    ["gemini"]="gemini"
    ["claude"]="claude"
    ["codex"]="codex"
)

MODE="update"
INSTALL_MISSING=0
TARGETS=()
NPM_GLOBAL_JSON=""

usage() {
    cat <<EOF
Usage:
  ${SCRIPT_NAME} [--check] [--install-missing] [gemini] [claude] [codex]

Examples:
  ${SCRIPT_NAME}
  ${SCRIPT_NAME} --check
  ${SCRIPT_NAME} codex
  ${SCRIPT_NAME} --install-missing gemini claude codex
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_cmd() {
    local cmd="$1"
    command -v "${cmd}" >/dev/null 2>&1 || die "missing required command: ${cmd}"
}

is_valid_target() {
    local target="$1"
    [[ -n "${PACKAGES[${target}]:-}" ]]
}

refresh_global_json() {
    NPM_GLOBAL_JSON="$(npm -g list --depth=0 --json 2>/dev/null || true)"
}

get_installed_version() {
    local package_name="$1"
    node -e '
const raw = process.argv[1];
const packageName = process.argv[2];
if (!raw) {
  process.exit(0);
}
try {
  const data = JSON.parse(raw);
  process.stdout.write(data.dependencies?.[packageName]?.version || "");
} catch (error) {
  process.exit(1);
}
' "${NPM_GLOBAL_JSON}" "${package_name}"
}

get_latest_version() {
    local package_name="$1"
    npm view "${package_name}" version 2>/dev/null || true
}

get_binary_path() {
    local binary_name="$1"
    command -v "${binary_name}" 2>/dev/null || true
}

print_status() {
    local target="$1"
    local package_name="${PACKAGES[${target}]}"
    local binary_name="${BINARIES[${target}]}"
    local installed_version
    local latest_version
    local binary_path
    local status

    installed_version="$(get_installed_version "${package_name}")"
    latest_version="$(get_latest_version "${package_name}")"
    binary_path="$(get_binary_path "${binary_name}")"

    if [[ -z "${installed_version}" ]]; then
        status="not-installed"
    elif [[ -n "${latest_version}" && "${installed_version}" == "${latest_version}" ]]; then
        status="up-to-date"
    elif [[ -n "${latest_version}" ]]; then
        status="needs-update"
    else
        status="installed"
    fi

    echo "=== ${target} ==="
    echo "package : ${package_name}"
    echo "binary  : ${binary_path:-<not found in PATH>}"
    echo "current : ${installed_version:-<not installed>}"
    echo "latest  : ${latest_version:-<unknown>}"
    echo "status  : ${status}"
    echo ""
}

update_target() {
    local target="$1"
    local package_name="${PACKAGES[${target}]}"
    local binary_name="${BINARIES[${target}]}"
    local before_version
    local latest_version
    local after_version

    before_version="$(get_installed_version "${package_name}")"
    latest_version="$(get_latest_version "${package_name}")"

    if [[ -z "${before_version}" && "${INSTALL_MISSING}" -ne 1 ]]; then
        echo "[skip] ${target}: package is not installed. Use --install-missing to install it."
        return 0
    fi

    if [[ -n "${before_version}" && -n "${latest_version}" && "${before_version}" == "${latest_version}" ]]; then
        echo "[ok] ${target}: already at latest version (${before_version})."
        return 0
    fi

    if [[ -z "${before_version}" ]]; then
        echo "[install] ${target}: installing ${package_name}@latest"
    else
        echo "[update] ${target}: ${before_version} -> ${latest_version:-latest}"
    fi

    npm install -g "${package_name}@latest"

    refresh_global_json
    after_version="$(get_installed_version "${package_name}")"
    [[ -n "${after_version}" ]] || die "failed to install/update ${package_name}"

    echo "[done] ${target}: ${before_version:-<not installed>} -> ${after_version}"
    echo "       binary: $(get_binary_path "${binary_name}")"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            MODE="check"
            ;;
        --install-missing)
            INSTALL_MISSING=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        all)
            TARGETS=("${DEFAULT_TARGETS[@]}")
            ;;
        gemini|claude|codex)
            TARGETS+=("$1")
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
    shift
done

require_cmd npm
require_cmd node

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("${DEFAULT_TARGETS[@]}")
fi

refresh_global_json

echo "Global npm prefix: $(npm prefix -g)"
echo ""

if [[ "${MODE}" == "check" ]]; then
    for target in "${TARGETS[@]}"; do
        is_valid_target "${target}" || die "unknown target: ${target}"
        print_status "${target}"
    done
    exit 0
fi

for target in "${TARGETS[@]}"; do
    is_valid_target "${target}" || die "unknown target: ${target}"
    update_target "${target}"
done

echo ""
echo "Final status:"
echo ""
refresh_global_json
for target in "${TARGETS[@]}"; do
    print_status "${target}"
done
