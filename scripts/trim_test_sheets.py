"""
trim_test_sheets.py — One-time pre-trim script.

PURPOSE
-------
The operational TEST Google Sheet (nse_stock_data) was never trimmed because
the pipeline's overwrite+cleanup step was always blocked by the Timestamp
serialization bug.  As a result the sheet still holds 2,800+ rows of raw OHLCV
data from 2015.

Running the full pipeline against 2,800-row sheets would:
  • Read  ~137K cells from Google Sheets (quota-heavy)
  • Run FE on 2,800 rows × 49 symbols  (very slow)
  • Write ~137K cells back                (quota-heavy)
  • Archive ~135K cells to training sheet (quota-heavy)

This one-time script trims each worksheet to the last KEEP_ROWS rows *without*
archiving — the historical data already exists in nse_stock_data_train.

After running this script, run the full pipeline as normal:
    docker compose --profile tools run --rm pipeline \
        python run_full_workflow.py --live 2>&1 | tee trim_$(date +%Y%m%dT%H%M%S).log

USAGE
-----
    python scripts/trim_test_sheets.py [--keep 60] [--worksheets RELIANCE,TCS] [--dry-run]

ENVIRONMENT
-----------
    GOOGLE_CREDENTIALS or GOOGLE_APPLICATION_CREDENTIALS — path to service-account JSON
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Set

SHEET_ID = "1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o"
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

DEFAULT_KEEP_ROWS = 60          # SMA_50 needs 51+; 60 gives a 9-row buffer
INTER_WORKSHEET_SLEEP = 2.0     # seconds between worksheets — stay inside quota


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trim nse_stock_data worksheets to last N rows.")
    p.add_argument("--keep", type=int, default=DEFAULT_KEEP_ROWS,
                   help=f"Number of most-recent rows to keep (default: {DEFAULT_KEEP_ROWS})")
    p.add_argument("--worksheets", default=None,
                   help="Comma-separated worksheet names to process (default: all)")
    p.add_argument("--sheet-id", default=SHEET_ID,
                   help="Google Sheets spreadsheet ID to trim")
    p.add_argument("--google-credentials", default=None,
                   help="Path to service-account JSON (or set GOOGLE_CREDENTIALS env var)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be trimmed without making any changes")
    return p.parse_args()


def authorize(credentials_path: Optional[str]):
    import gspread
    from google.oauth2.service_account import Credentials

    # Resolve candidate paths in preference order:
    #   1. Explicit --google-credentials argument
    #   2. Project-local credentials (same directory as this script)
    #   3. GOOGLE_CREDENTIALS env var
    #   4. GOOGLE_APPLICATION_CREDENTIALS env var (last — may point to a
    #      different project on this machine)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_local = os.path.join(script_dir, "credentials", "Credentials_New.json")

    candidates = [
        credentials_path,
        project_local if os.path.exists(project_local) else None,
        os.environ.get("GOOGLE_CREDENTIALS"),
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    ]
    cred_file = next((c for c in candidates if c and os.path.isfile(c)), None)

    if not cred_file:
        checked = [c for c in candidates if c]
        raise RuntimeError(
            "No valid credentials file found. Checked:\n"
            + "\n".join(f"  {p}" for p in checked)
            + "\nPass --google-credentials <path> or set GOOGLE_CREDENTIALS env var."
        )

    print(f"  Using credentials: {cred_file}", flush=True)
    creds = Credentials.from_service_account_file(cred_file, scopes=GOOGLE_SCOPES)
    return gspread.authorize(creds)


def trim_worksheet(worksheet, keep: int, dry_run: bool) -> dict:
    """Keep only the last `keep` data rows; delete the rest from the top."""
    title = worksheet.title
    values = worksheet.get_all_values()
    if not values:
        return {"worksheet": title, "status": "empty", "rows_before": 0, "rows_after": 0, "deleted": 0}

    header = values[0]
    data   = values[1:]
    total  = len(data)

    if total <= keep:
        return {
            "worksheet": title, "status": "no_trim_needed",
            "rows_before": total, "rows_after": total, "deleted": 0,
        }

    # Keep the LAST `keep` rows
    to_keep   = data[-keep:]
    n_deleted = total - keep

    if dry_run:
        print(f"  [DRY-RUN] {title}: would delete {n_deleted} rows (keep last {keep})", flush=True)
        return {
            "worksheet": title, "status": "dry_run",
            "rows_before": total, "rows_after": keep, "deleted": n_deleted,
        }

    # Clear the sheet and rewrite header + last `keep` rows
    new_values = [header] + to_keep
    worksheet.clear()
    time.sleep(0.5)  # brief pause after clear before update
    worksheet.update(new_values, "A1")
    print(f"  {title}: trimmed {total} → {keep} rows (deleted {n_deleted})", flush=True)
    return {
        "worksheet": title, "status": "trimmed",
        "rows_before": total, "rows_after": keep, "deleted": n_deleted,
    }


def parse_csv(text: Optional[str]) -> Optional[Set[str]]:
    if not text:
        return None
    return {part.strip().upper() for part in text.split(",") if part.strip()}


def main() -> None:
    args = parse_args()
    requested: Optional[Set[str]] = parse_csv(args.worksheets)

    print(f"Connecting to Google Sheets …", flush=True)
    client = authorize(args.google_credentials)
    spreadsheet = client.open_by_key(args.sheet_id)
    worksheets = spreadsheet.worksheets()

    if requested:
        worksheets = [ws for ws in worksheets if ws.title.upper() in requested]

    total = len(worksheets)
    print(
        f"{'[DRY-RUN] ' if args.dry_run else ''}Trimming {total} worksheet(s) "
        f"to last {args.keep} rows …\n",
        flush=True,
    )

    results: List[dict] = []
    for idx, ws in enumerate(worksheets, start=1):
        print(f"[{idx}/{total}] {ws.title} …", end=" ", flush=True)
        try:
            result = trim_worksheet(ws, args.keep, args.dry_run)
            results.append(result)
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)
            results.append({"worksheet": ws.title, "status": "error", "error": str(exc)})

        if idx < total:
            time.sleep(INTER_WORKSHEET_SLEEP)

    # Summary
    print("\n── Summary ──────────────────────────────────")
    trimmed   = sum(1 for r in results if r.get("status") == "trimmed")
    dry_run_c = sum(1 for r in results if r.get("status") == "dry_run")
    skipped   = sum(1 for r in results if r.get("status") == "no_trim_needed")
    errors    = sum(1 for r in results if r.get("status") == "error")
    deleted   = sum(r.get("deleted", 0) for r in results)

    if args.dry_run:
        print(f"  Would trim : {dry_run_c} worksheet(s)  ({deleted} rows would be deleted)")
        print(f"  Would skip : {skipped} worksheet(s)  (already ≤ {args.keep} rows)")
    else:
        print(f"  Trimmed    : {trimmed} worksheet(s)  ({deleted} rows deleted)")
        print(f"  Skipped    : {skipped} worksheet(s)  (already ≤ {args.keep} rows)")
    if errors:
        print(f"  Errors     : {errors} worksheet(s)")
        for r in results:
            if r.get("status") == "error":
                print(f"    {r['worksheet']}: {r.get('error')}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
