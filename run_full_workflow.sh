#!/usr/bin/env bash
# =============================================================================
# run_full_workflow.sh
# -----------------------------------------------------------------------------
# Safe launcher for the master end-to-end orchestrator (run_full_workflow.py).
#
# Responsibilities:
#   * resolve and activate a Python environment (venv if present)
#   * export the environment variables the pipeline needs (credentials, sheet
#     IDs, paths) so the workflow is self-configuring
#   * invoke run_full_workflow.py and tee all output to a timestamped log
#   * fail safely with a clear, non-zero exit code on any error
#
# Usage:
#   ./run_full_workflow.sh                      # dry-run (default, no writes)
#   ./run_full_workflow.sh --live               # real end-to-end run
#   ./run_full_workflow.sh --worksheets RELIANCE,TCS
#   ./run_full_workflow.sh --live --worksheets RELIANCE
#
# Exit codes:  0 success / success_with_warnings   1 failure
# =============================================================================

set -Eeuo pipefail

# ── Resolve project root (this script's directory) ──────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# ── Colours / logging helpers ───────────────────────────────────────────────
if [[ -t 1 ]]; then
  RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YLW=$'\033[1;33m'
  CYN=$'\033[0;36m'; RST=$'\033[0m'
else
  RED=''; GRN=''; YLW=''; CYN=''; RST=''
fi
info()  { echo "${CYN}[launcher]${RST} $*"; }
ok()    { echo "${GRN}[launcher]${RST} $*"; }
warn()  { echo "${YLW}[launcher]${RST} $*" >&2; }
err()   { echo "${RED}[launcher]${RST} $*" >&2; }

# ── Trap: report the exact failing command ──────────────────────────────────
on_error() {
  local exit_code=$?
  err "FAILED (exit ${exit_code}) at line ${BASH_LINENO[0]}: ${BASH_COMMAND}"
  exit "${exit_code}"
}
trap on_error ERR

# ── Log file (timestamped, never overwritten) ───────────────────────────────
LOG_DIR="${PROJECT_ROOT}/logs/workflow"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%dT%H%M%SZ)"
LAUNCH_LOG="${LOG_DIR}/launch_${TS}.log"
info "Launcher log: ${LAUNCH_LOG}"

# ── Resolve a Python interpreter ────────────────────────────────────────────
# Priority: explicit $PYTHON_BIN -> project venv -> Homebrew -> system python3
#
# We must *verify* each candidate actually executes — on macOS, /usr/bin/python3
# (and any venv symlink pointing to it) is an Xcode stub that exits 72 and
# triggers a GUI installer prompt instead of running Python.
python_works() {
  # Returns 0 only when the binary runs a trivial script without error.
  local bin="$1"
  [[ -x "${bin}" ]] || return 1
  "${bin}" -c "import sys; sys.exit(0)" >/dev/null 2>&1
}

PYTHON_BIN="${PYTHON_BIN:-}"

# 1. If the caller pre-set PYTHON_BIN, honour it — but still validate it.
if [[ -n "${PYTHON_BIN}" ]]; then
  if ! python_works "${PYTHON_BIN}"; then
    warn "Caller-supplied PYTHON_BIN='${PYTHON_BIN}' does not work; searching for another."
    PYTHON_BIN=""
  fi
fi

# 2. Search venv, then Homebrew, then $PATH (in order of preference).
if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in \
      "${PROJECT_ROOT}/venv/bin/python3" \
      "${PROJECT_ROOT}/venv/bin/python" \
      "${PROJECT_ROOT}/.venv/bin/python3" \
      "${PROJECT_ROOT}/.venv/bin/python" \
      "/opt/homebrew/bin/python3" \
      "/usr/local/bin/python3" \
      "/opt/homebrew/bin/python3.13" \
      "/opt/homebrew/bin/python3.12" \
      "/opt/homebrew/bin/python3.11" \
      "/usr/local/bin/python3.13" \
      "/usr/local/bin/python3.12" \
      "/usr/local/bin/python3.11"; do
    if python_works "${candidate}"; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

# 3. Last resort: whatever python3/python is on PATH (skipping stubs).
if [[ -z "${PYTHON_BIN}" ]]; then
  for cmd in python3 python3.13 python3.12 python3.11 python; do
    _resolved="$(command -v "${cmd}" 2>/dev/null || true)"
    if [[ -n "${_resolved}" ]] && python_works "${_resolved}"; then
      PYTHON_BIN="${_resolved}"
      break
    fi
  done
  unset _resolved
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  err "No working Python interpreter found."
  err "Install Python 3 via Homebrew:  brew install python3"
  exit 1
fi

# Activate the venv environment if we resolved one inside the project (best-effort).
VENV_ACTIVATE="$(dirname "${PYTHON_BIN}")/activate"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}" || warn "Could not source ${VENV_ACTIVATE} (continuing)."
  ok "Activated environment: $(dirname "$(dirname "${PYTHON_BIN}")")"
fi
info "Python interpreter: ${PYTHON_BIN}"
# Print version non-fatally — the stub check above already guarantees Python works.
"${PYTHON_BIN}" --version 2>&1 | sed 's/^/[launcher] /' || true

# ── Load .env if present (without clobbering already-exported vars) ─────────
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  info "Loading environment from .env"
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env" 2>/dev/null || warn ".env partially loaded."
  set +a
fi

# ── Required environment variables (sane defaults) ──────────────────────────
export BASE_DIR="${BASE_DIR:-${PROJECT_ROOT}}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

# Credentials — default to the path supplied by the project.
DEFAULT_CRED="${PROJECT_ROOT}/credentials/Credentials_New.json"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-${DEFAULT_CRED}}"
export GOOGLE_CREDENTIALS="${GOOGLE_CREDENTIALS:-${GOOGLE_APPLICATION_CREDENTIALS}}"

# Spreadsheet IDs (TEST = operational, TRAIN = historical archive).
export SHEET_ID="${SHEET_ID:-1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o}"
export OPERATIONAL_SHEET_ID="${OPERATIONAL_SHEET_ID:-${SHEET_ID}}"
export HISTORICAL_TRAINING_SHEET_ID="${HISTORICAL_TRAINING_SHEET_ID:-1_gpRclY80tH3r54b9R5HTXqKF7R7bnMoWugF9Fy-boI}"

# Pipeline paths.
export METADATA_PATH="${METADATA_PATH:-${PROJECT_ROOT}/outputs/pipeline_metadata.json}"
export MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/outputs/Saved_Models}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/main_inference}"
export DEVICE="${DEVICE:-auto}"

# ── Pre-flight checks ───────────────────────────────────────────────────────
if [[ ! -f "${PROJECT_ROOT}/run_full_workflow.py" ]]; then
  err "run_full_workflow.py not found in ${PROJECT_ROOT}"
  exit 1
fi
if [[ ! -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]]; then
  warn "Credentials file not found: ${GOOGLE_APPLICATION_CREDENTIALS}"
  warn "Dry-run can still proceed; --live runs will fail startup validation."
fi
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/state" \
         "${PROJECT_ROOT}/outputs/main_inference"

# ── Announce mode ───────────────────────────────────────────────────────────
if [[ " $* " == *" --live "* ]]; then
  warn "LIVE MODE — this run WILL modify your Google Sheets."
else
  info "DRY-RUN mode (default) — no Google Sheet will be modified."
fi

# ── Execute the orchestrator ────────────────────────────────────────────────
info "Starting orchestrator: ${PYTHON_BIN} run_full_workflow.py $*"
set +e
"${PYTHON_BIN}" "${PROJECT_ROOT}/run_full_workflow.py" "$@" 2>&1 | tee -a "${LAUNCH_LOG}"
WORKFLOW_EXIT="${PIPESTATUS[0]}"
set -e

# ── Report ──────────────────────────────────────────────────────────────────
echo
if [[ "${WORKFLOW_EXIT}" -eq 0 ]]; then
  ok "Workflow completed successfully (exit ${WORKFLOW_EXIT})."
  ok "Summary: outputs/workflow/run_summary_latest.json"
else
  err "Workflow FAILED (exit ${WORKFLOW_EXIT})."
  err "Inspect: ${LAUNCH_LOG}"
  err "Failures log: ${LOG_DIR}/failures_*.log"
fi
info "Full launcher log saved to: ${LAUNCH_LOG}"
exit "${WORKFLOW_EXIT}"
