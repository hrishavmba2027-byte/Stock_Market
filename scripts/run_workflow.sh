#!/usr/bin/env bash
# =============================================================================
# run_workflow.sh — Stock Market Automation workflow runner
#
# Usage:
#   ./run_workflow.sh                        # full run, all stocks
#   ./run_workflow.sh --stock RELIANCE       # single stock
#   ./run_workflow.sh --stock "RELIANCE,TCS" # multiple stocks
#   ./run_workflow.sh --force                # force even if no sheet changes
#   ./run_workflow.sh --dry-run              # validate without writing
#   ./run_workflow.sh --cli                  # run via Docker CLI (not API)
#   ./run_workflow.sh --local                # run directly with local Python
#   ./run_workflow.sh status                 # print last-run status and exit
#   ./run_workflow.sh health                 # health-check only
#   ./run_workflow.sh logs                   # tail live container logs
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
API_URL="${API_BASE_URL:-http://localhost:8000}"
TIMEOUT=1800          # seconds before giving up waiting for a run to finish
POLL_INTERVAL=5       # seconds between status polls
COMPOSE_PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="${COMPOSE_PROJECT_DIR}/venv/bin/python"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
section() { echo -e "\n${BOLD}── $* ──${RESET}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
MODE="api"           # api | cli | local
STOCK=""
FORCE=false
DRY_RUN=false
SUBCOMMAND="${1:-run}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    run|status|health|logs) SUBCOMMAND="$1"; shift ;;
    --stock|-s)    STOCK="$2";  shift 2 ;;
    --force|-f)    FORCE=true;  shift ;;
    --dry-run|-d)  DRY_RUN=true; shift ;;
    --cli)         MODE="cli";  shift ;;
    --local)       MODE="local"; shift ;;
    --api)         MODE="api";  shift ;;
    --help|-h)
      sed -n '/^# Usage/,/^# ===/p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) warn "Unknown argument: $1"; shift ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
require_jq() {
  if ! command -v jq &>/dev/null; then
    warn "jq not installed — raw JSON will be shown. Install with: brew install jq"
    JQ="cat"
  else
    JQ="jq"
  fi
}

pretty_json() {
  if command -v jq &>/dev/null; then jq '.'; else cat; fi
}

check_containers_running() {
  if ! docker compose -f "${COMPOSE_PROJECT_DIR}/docker-compose.yml" ps --format json 2>/dev/null \
      | grep -q '"State":"running"'; then
    error "Containers are not running."
    echo ""
    echo "  Start them first with:"
    echo "    docker compose up -d"
    exit 1
  fi
}

wait_for_api() {
  info "Waiting for API at ${API_URL}/health ..."
  local attempts=0
  until curl -sf "${API_URL}/health" >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if [[ $attempts -ge 30 ]]; then
      error "API did not become healthy after 30 attempts."
      exit 1
    fi
    sleep 2
  done
  ok "API is healthy."
}

# ── Sub-commands ──────────────────────────────────────────────────────────────

cmd_health() {
  section "Health Check"
  wait_for_api
  curl -sf "${API_URL}/health" | pretty_json
}

cmd_status() {
  section "Last Run Status"
  wait_for_api
  curl -sf "${API_URL}/status" | pretty_json
}

cmd_logs() {
  section "Container Logs (Ctrl-C to exit)"
  docker compose -f "${COMPOSE_PROJECT_DIR}/docker-compose.yml" logs -f
}

# ── API mode: POST /run ───────────────────────────────────────────────────────
cmd_run_api() {
  section "Triggering Workflow via API"
  check_containers_running
  wait_for_api

  # Build JSON payload
  # NOTE: Never interpolate shell booleans/variables directly into Python source
  # as Python literals — bash `false` ≠ Python `False`.  Instead, pass all
  # dynamic values as environment variables and decode them inside Python.
  local worksheets_csv="${STOCK:-}"
  local force_flag="${FORCE}"   # bash string: "true" or "false"

  local payload
  payload=$(
    WORKSHEETS_CSV="${worksheets_csv}" \
    FORCE_FLAG="${force_flag}" \
    python3 - <<'PYEOF'
import json, os

csv = os.environ.get("WORKSHEETS_CSV", "").strip()
worksheets = [s.strip() for s in csv.split(",") if s.strip()] if csv else []
force = os.environ.get("FORCE_FLAG", "false").lower() == "true"

print(json.dumps({
    "worksheets": worksheets,
    "force": force,
    "reason": "manual_run_script",
}))
PYEOF
  )

  info "Payload: ${payload}"
  info "POST ${API_URL}/run"

  local response
  response=$(curl -sf -X POST "${API_URL}/run" \
    -H "Content-Type: application/json" \
    -d "${payload}" \
    --max-time "${TIMEOUT}") || {
      error "API call failed (curl exit $?). Is the container healthy?"
      exit 1
    }

  echo ""
  section "Workflow Result"
  echo "$response" | pretty_json

  # Extract status for exit code
  local status
  status=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "unknown")

  echo ""
  case "$status" in
    success) ok  "Workflow completed successfully (status=success)" ;;
    skipped) warn "Workflow was skipped (no new data or not forced)" ;;
    error)   error "Workflow finished with errors (status=error)"; exit 1 ;;
    *)       warn "Unknown status: ${status}" ;;
  esac
}

# ── CLI mode: docker compose run --rm ────────────────────────────────────────
cmd_run_cli() {
  section "Running Workflow via Docker CLI"

  local args=("run" "--rm" "api" "python" "app_data.py" "run")

  [[ -n "$STOCK" ]]  && args+=("--worksheets" "$STOCK")
  [[ "$FORCE" == true ]]   && args+=("--reason" "force_run")
  [[ "$DRY_RUN" == true ]] && args+=("--dry-run")

  info "Command: docker compose ${args[*]}"
  docker compose -f "${COMPOSE_PROJECT_DIR}/docker-compose.yml" "${args[@]}"
}

# ── Local mode: use venv Python directly ─────────────────────────────────────
cmd_run_local() {
  section "Running Workflow Locally (venv)"

  if [[ ! -x "$VENV_PYTHON" ]]; then
    error "venv Python not found at ${VENV_PYTHON}"
    echo "  Activate your venv first:  source venv/bin/activate"
    exit 1
  fi

  local args=("${VENV_PYTHON}" "app_data.py" "run")

  [[ -n "$STOCK" ]]  && args+=("--worksheets" "$STOCK")
  [[ "$DRY_RUN" == true ]] && args+=("--dry-run")

  info "Command: ${args[*]}"
  cd "${COMPOSE_PROJECT_DIR}" && "${args[@]}"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
require_jq

case "$SUBCOMMAND" in
  health) cmd_health ;;
  status) cmd_status ;;
  logs)   cmd_logs ;;
  run)
    case "$MODE" in
      api)   cmd_run_api ;;
      cli)   cmd_run_cli ;;
      local) cmd_run_local ;;
    esac
    ;;
esac
