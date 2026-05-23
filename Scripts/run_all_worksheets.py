#!/usr/bin/env python3
"""
run_all_worksheets.py — Master orchestrator for the LIVE forecasting workflow.

Execution order per run
-----------------------
  STEP 1  Read all worksheet names from the LIVE Google Sheet.
  STEP 2  Refresh / update the live sheet data via Data_update.py.
  STEP 3  Run Feature_Engineering.py ONCE (must finish before any forecasting).
  STEP 4  Forecast every worksheet in a bounded thread-pool (main.py, parallel).
  STEP 5  Print a concise execution summary to stdout.

Design constraints (strictly followed)
---------------------------------------
  * Only this file is modified.
  * No forecasting / feature-engineering logic is duplicated here.
  * Individual child scripts are called via subprocess — their internal logic
    is fully preserved.
  * ThreadPoolExecutor is used for controlled parallel worksheet execution.
  * Exponential-backoff retry wraps every subprocess call.
  * A failed worksheet never terminates the rest of the workflow.
  * Only the LIVE sheet (LIVE_SHEET_ID) is processed — never the historical
    training sheet.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from google.oauth2.service_account import Credentials
import gspread
import subprocess

# ── Paths ─────────────────────────────────────────────────────────────────────
# Scripts/ sits one level below the project root; resolve everything from here
# so the orchestrator works regardless of the caller's cwd.
_SCRIPTS_DIR  = Path(__file__).resolve().parent   # …/Stock_Market/Scripts/
_PROJECT_ROOT = _SCRIPTS_DIR.parent               # …/Stock_Market/

# ── Static configuration ───────────────────────────────────────────────────────
# Credential path: GOOGLE_APPLICATION_CREDENTIALS env var is the primary
# controller.  Falls back to the repo-relative default only when the variable
# is not set, so the same script works in both local and CI/CD environments
# without any code changes.
_DEFAULT_CRED = str(_PROJECT_ROOT / 'credentials' / 'Credentials_New.json')
CRED = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') \
    or os.environ.get('GOOGLE_CREDENTIALS') \
    or _DEFAULT_CRED
SCOPES         = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive',
]
LIVE_SHEET_ID  = '1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o'

# Child scripts (absolute paths — safe to call from any cwd)
_SCRIPT_DATA_UPDATE        = str(_PROJECT_ROOT / 'Data_update.py')
_SCRIPT_FEATURE_ENGINEERING = str(_PROJECT_ROOT / 'Feature_Engineering.py')
_SCRIPT_MAIN               = str(_PROJECT_ROOT / 'main.py')

# ── Tuning knobs ───────────────────────────────────────────────────────────────
MAX_WORKERS        = 3     # parallel forecast workers; keeps Sheets API happy
SUBMIT_STAGGER_SEC = 2     # seconds between worker submissions (burst protection)
SUBPROCESS_TIMEOUT = 600   # seconds; applies to each individual subprocess call
MAX_RETRIES        = 3     # max attempts per subprocess before marking as failed
RETRY_BASE_SEC     = 5     # exponential-backoff seed: sleep = RETRY_BASE_SEC * 2^(n-1)


# ── Environment builder ────────────────────────────────────────────────────────

def _build_env() -> Dict[str, str]:
    """
    Return a fresh copy of the current process environment with the required
    credential / config variables injected.  A copy is used so parallel workers
    never mutate shared state.
    """
    env = os.environ.copy()
    # Ensure both env vars are set to the resolved credential path so that
    # every child process (Data_update, Feature_Engineering, main.py) picks
    # up the correct file regardless of which variable they read first.
    env['GOOGLE_APPLICATION_CREDENTIALS'] = CRED
    env['GOOGLE_CREDENTIALS']             = CRED
    env['FORECAST_DAYS']                  = '15'
    print(f'[orchestrator] GOOGLE_APPLICATION_CREDENTIALS = {CRED}', flush=True)
    return env


# ── Core subprocess helper ─────────────────────────────────────────────────────

def _run_with_retry(
    cmd: List[str],
    label: str,
    env: Dict[str, str],
    max_retries: int = MAX_RETRIES,
) -> Tuple[int, int, str]:
    """
    Execute *cmd* with up to *max_retries* attempts and exponential backoff.

    Returns
    -------
    (exit_code, attempts_used, last_error_description)
      exit_code == 0 means success.
    """
    last_error = ''
    for attempt in range(1, max_retries + 1):
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                cwd=str(_PROJECT_ROOT),   # child scripts expect project root as cwd
                timeout=SUBPROCESS_TIMEOUT,
            )
            if proc.returncode == 0:
                return 0, attempt, ''
            last_error = f'non-zero exit ({proc.returncode})'

        except subprocess.TimeoutExpired:
            last_error = f'timeout after {SUBPROCESS_TIMEOUT}s'
        except OSError as exc:
            last_error = f'OS error: {exc}'
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

        if attempt < max_retries:
            sleep_sec = RETRY_BASE_SEC * (2 ** (attempt - 1))
            print(
                f'  [retry] {label}: attempt {attempt}/{max_retries} failed '
                f'({last_error}). Retrying in {sleep_sec}s…',
                flush=True,
            )
            time.sleep(sleep_sec)

    return 1, max_retries, last_error


# ── STEP 2: Refresh live sheet ─────────────────────────────────────────────────

def _refresh_live_sheet(env: Dict[str, str]) -> None:
    """
    Call Data_update.py to pull the latest market data into the live Google Sheet.
    A failure here is non-fatal: the orchestrator continues with the data already
    present in the sheet.
    """
    print('\n[STEP 2] Refreshing live Google Sheet data via Data_update.py …', flush=True)
    cmd = [
        sys.executable, _SCRIPT_DATA_UPDATE,
        '--sheet-id',           LIVE_SHEET_ID,
        '--google-credentials', CRED,
    ]
    rc, attempts, err = _run_with_retry(cmd, 'Data_update.py', env)
    if rc != 0:
        print(
            f'  WARNING: Data_update.py failed after {attempts} attempt(s): {err}\n'
            '  Continuing with existing sheet data.',
            flush=True,
        )
    else:
        print(f'  Data refresh complete (attempt {attempts}).', flush=True)


# ── STEP 3: Feature engineering ────────────────────────────────────────────────

def _run_feature_engineering(env: Dict[str, str]) -> bool:
    """
    Call Feature_Engineering.py ONCE.  This MUST complete before any main.py
    invocation begins, because the model depends on the engineered feature columns
    that this script writes to the workbook.

    Returns True if the script succeeded.
    """
    print('\n[STEP 3] Running Feature_Engineering.py (before any forecasting) …', flush=True)
    cmd = [sys.executable, _SCRIPT_FEATURE_ENGINEERING]
    rc, attempts, err = _run_with_retry(cmd, 'Feature_Engineering.py', env)
    if rc != 0:
        print(
            f'  WARNING: Feature_Engineering.py failed after {attempts} attempt(s): {err}\n'
            '  main.py embeds its own feature-engineering imports and will proceed,\n'
            '  but pre-computed workbook features may be stale.',
            flush=True,
        )
        return False
    print(f'  Feature engineering complete (attempt {attempts}).', flush=True)
    return True


# ── STEP 4/5: Per-worksheet forecasting ───────────────────────────────────────

def _forecast_worksheet(
    ws: str,
    env: Dict[str, str],
) -> Tuple[str, bool, int, str]:
    """
    Worker function executed inside the thread pool.

    Calls main.py for a single worksheet with:
      --all-eligible-rows          → process every eligible row
      --refresh-existing-forecasts → overwrite any existing forecast columns

    Returns (worksheet_name, success, attempts_used, error_description).
    """
    cmd = [
        sys.executable, _SCRIPT_MAIN,
        '--google-credentials',      CRED,
        '--sheet-id',                LIVE_SHEET_ID,
        '--worksheet',               ws,
        '--all-eligible-rows',
        '--refresh-existing-forecasts',
    ]
    rc, attempts, err = _run_with_retry(cmd, ws, env, max_retries=MAX_RETRIES)
    return ws, rc == 0, attempts, err


def _run_parallel_forecasting(
    worksheets: List[str],
    env: Dict[str, str],
) -> Dict[str, Dict]:
    """
    Submit all worksheets to a bounded ThreadPoolExecutor and collect results.

    * MAX_WORKERS caps concurrent API calls to Google Sheets.
    * SUBMIT_STAGGER_SEC introduces a short pause between submissions to avoid
      API burst errors at the start of a run.
    * Each worker gets its OWN env copy so there is no shared mutable state
      between threads.
    * Results are recorded as futures complete (not in submission order) so a
      slow worksheet does not block the summary.

    Returns a dict keyed by worksheet name:
      {'success': bool, 'attempts': int, 'error': str}
    """
    print(
        f'\n[STEP 4/5] Forecasting {len(worksheets)} worksheet(s) '
        f'with up to {MAX_WORKERS} parallel workers …',
        flush=True,
    )

    results: Dict[str, Dict] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_ws = {}
        for i, ws in enumerate(worksheets):
            if i > 0:
                # Stagger submissions to prevent burst-hitting the Sheets API
                time.sleep(SUBMIT_STAGGER_SEC)
            # Pass a *copy* of env to each worker — avoids any cross-thread mutation
            future = pool.submit(_forecast_worksheet, ws, env.copy())
            future_to_ws[future] = ws
            print(f'  → submitted: {ws}', flush=True)

        for future in as_completed(future_to_ws):
            ws_name, success, attempts, err = future.result()
            results[ws_name] = {
                'success':  success,
                'attempts': attempts,
                'error':    err,
            }
            status_glyph = '✓' if success else '✗'
            detail = f' — {err}' if not success else ''
            print(
                f'  {status_glyph} {ws_name}  (attempts={attempts}){detail}',
                flush=True,
            )

    return results


# ── Final summary ──────────────────────────────────────────────────────────────

def _print_summary(
    worksheets: List[str],
    results: Dict[str, Dict],
    fe_succeeded: bool,
) -> None:
    total     = len(worksheets)
    succeeded = sum(1 for r in results.values() if r['success'])
    failed    = total - succeeded
    retried   = sum(1 for r in results.values() if r['attempts'] > 1)

    sep = '=' * 62
    print(f'\n{sep}')
    print('  LIVE FORECASTING — EXECUTION SUMMARY')
    print(sep)
    print(f'  Sheet ID              : {LIVE_SHEET_ID}')
    print(f'  Feature engineering   : {"OK" if fe_succeeded else "FAILED (main.py fallback used)"}')
    print(f'  Total worksheets      : {total}')
    print(f'  Successful            : {succeeded}')
    print(f'  Failed                : {failed}')
    print(f'  Worksheets retried    : {retried}')

    if failed:
        print(f'\n  Failed worksheets:')
        for ws, r in results.items():
            if not r['success']:
                print(f'    ✗  {ws}  (attempts={r["attempts"]}, error={r["error"]})')

    print(sep)
    print('  Forecast overwrite    : ENABLED (--refresh-existing-forecasts)')
    print('  Parallel execution    : ENABLED (ThreadPoolExecutor, '
          f'max_workers={MAX_WORKERS})')
    print('  Retry logic           : ENABLED '
          f'(max={MAX_RETRIES} attempts, exponential backoff, base={RETRY_BASE_SEC}s)')
    print('  Source sheet          : LIVE only — historical training sheet NOT touched')
    print(f'{sep}\n')


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # ------------------------------------------------------------------
    # STEP 1 — Read all worksheet names from the LIVE Google Sheet ONLY.
    # ------------------------------------------------------------------
    print('[STEP 1] Reading worksheet list from LIVE Google Sheet …', flush=True)
    try:
        creds  = Credentials.from_service_account_file(CRED, scopes=SCOPES)
        client = gspread.authorize(creds)
        sh     = client.open_by_key(LIVE_SHEET_ID)
        # Deduplicate while preserving order (guards against API oddities)
        seen: set = set()
        worksheets: List[str] = []
        for w in sh.worksheets():
            if w.title not in seen:
                seen.add(w.title)
                worksheets.append(w.title)
    except Exception as exc:
        print(f'FATAL: Could not read worksheet list — {exc}', flush=True)
        sys.exit(1)

    if not worksheets:
        print('No worksheets found in the LIVE sheet. Exiting.', flush=True)
        sys.exit(0)

    print(f'  Found {len(worksheets)} worksheet(s): {worksheets}', flush=True)

    # Build the shared base environment (each worker will receive a copy)
    base_env = _build_env()

    # ------------------------------------------------------------------
    # STEP 2 — Refresh live sheet data.
    # ------------------------------------------------------------------
    _refresh_live_sheet(base_env)

    # ------------------------------------------------------------------
    # STEP 3 — Feature engineering (MUST complete before any forecasting).
    # ------------------------------------------------------------------
    fe_ok = _run_feature_engineering(base_env)

    # ------------------------------------------------------------------
    # STEPS 4 + 5 — Parallel forecasting with retry; overwrite forecasts.
    # ------------------------------------------------------------------
    results = _run_parallel_forecasting(worksheets, base_env)

    # ------------------------------------------------------------------
    # Final summary (terminal only — no log files created).
    # ------------------------------------------------------------------
    _print_summary(worksheets, results, fe_ok)


if __name__ == '__main__':
    main()
