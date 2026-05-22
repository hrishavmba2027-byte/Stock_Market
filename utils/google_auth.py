"""
utils/google_auth.py — Centralized Google authentication for the Stock Market repo.

All scripts, services, and modules should import from here instead of
repeating auth boilerplate:

    from utils.google_auth import get_gspread_client, get_credentials

Credential resolution priority (highest → lowest):
  1. Explicit ``credentials_path`` argument passed to the function
  2. ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable  ← primary standard
  3. ``GOOGLE_CREDENTIALS`` environment variable              ← legacy fallback
  4. ``<project_root>/credentials/Credentials_New.json``      ← last-resort default

Set the primary variable in your shell or .env:
    export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/Credentials_New.json

The module:
  * Uses google.oauth2.service_account.Credentials (modern, JWT-safe).
  * Prints debug info every time auth is initialized.
  * Includes pre-flight validation (file exists, Sheet accessible, write ok).
  * Never swallows errors silently — all failures raise with clear messages.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

# ── Path resolution ────────────────────────────────────────────────────────────
# utils/google_auth.py lives at <project_root>/utils/google_auth.py
_UTILS_DIR    = Path(__file__).resolve().parent   # …/Stock_Market/utils/
_PROJECT_ROOT = _UTILS_DIR.parent                 # …/Stock_Market/

# Fallback credential path (used only when no env var is set).
_DEFAULT_CREDS_PATH = _PROJECT_ROOT / "credentials" / "Credentials_New.json"

# Google API scopes required for Sheets + Drive access.
SCOPES: List[str] = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# ── Path resolver ──────────────────────────────────────────────────────────────

def resolve_credentials_path(override: Optional[str] = None) -> Path:
    """
    Return the absolute path to the service-account JSON.

    Resolution order
    ----------------
    1. ``override`` argument (e.g. from a CLI ``--google-credentials`` flag)
    2. ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable  ← primary
    3. ``GOOGLE_CREDENTIALS`` environment variable              ← legacy
    4. ``<project_root>/credentials/Credentials_New.json``      ← last resort

    Raises ``FileNotFoundError`` if none of the candidates point to an
    existing file, with a message showing exactly what was tried.
    """
    candidates = [
        override,
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),   # primary standard
        os.environ.get("GOOGLE_CREDENTIALS"),               # legacy fallback
        str(_DEFAULT_CREDS_PATH),                           # last-resort default
    ]

    for candidate in candidates:
        if candidate:
            p = Path(candidate).expanduser()
            # Resolve relative paths against project root so scripts called
            # from any cwd still find the file.
            if not p.is_absolute():
                p = (_PROJECT_ROOT / p).resolve()
            if p.exists():
                return p

    # None of the candidates existed — surface a clear, actionable error.
    tried = [c for c in candidates if c]
    raise FileNotFoundError(
        "Google credentials file not found.\n\n"
        "Set the environment variable and re-run:\n"
        "  export GOOGLE_APPLICATION_CREDENTIALS="
        "/absolute/path/to/Credentials_New.json\n\n"
        "Tried:\n" + "\n".join(f"  {t}" for t in tried)
    )


# ── Core auth factory ──────────────────────────────────────────────────────────

def get_credentials(
    credentials_path: Optional[str] = None,
    scopes: Optional[List[str]] = None,
) -> Any:
    """
    Load and return a ``google.oauth2.service_account.Credentials`` object.

    Parameters
    ----------
    credentials_path : str, optional
        Explicit path to the service-account JSON.  When omitted the path is
        resolved via :func:`resolve_credentials_path`.
    scopes : list of str, optional
        OAuth scopes.  Defaults to :data:`SCOPES` (Sheets + Drive).

    Returns
    -------
    google.oauth2.service_account.Credentials
    """
    from google.oauth2.service_account import Credentials  # lazy import

    resolved = resolve_credentials_path(credentials_path)
    effective_scopes = scopes or SCOPES

    # ── Debug logging ──────────────────────────────────────────────────────────
    print(f"[google_auth] Using credentials : {resolved}", flush=True)

    # Peek at the JSON to surface the service-account email for confirmation.
    try:
        with open(resolved) as fh:
            sa_info = json.load(fh)
        sa_email  = sa_info.get("client_email", "<unknown>")
        project   = sa_info.get("project_id",   "<unknown>")
        print(f"[google_auth] Service Account   : {sa_email}", flush=True)
        print(f"[google_auth] GCP Project        : {project}",   flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[google_auth] WARNING: could not inspect credentials JSON: {exc}", flush=True)

    creds = Credentials.from_service_account_file(str(resolved), scopes=effective_scopes)
    print("[google_auth] Service Account Loaded Successfully", flush=True)
    return creds


def _triage_auth_error(exc: Exception) -> None:
    """
    Print an actionable root-cause analysis for known Google auth errors.
    Called when get_credentials() or get_gspread_client() catches an exception.
    """
    msg = str(exc).lower()
    print(f"\n[google_auth] ── Auth Error Triage ────────────────────────────", flush=True)
    print(f"[google_auth] Error: {exc}", flush=True)

    if "invalid_grant" in msg and ("jwt" in msg or "signature" in msg):
        print("[google_auth] ROOT CAUSE: invalid_grant — Invalid JWT Signature", flush=True)
        print("[google_auth]", flush=True)
        print("[google_auth] This means Google received your JWT but could NOT verify its", flush=True)
        print("[google_auth] signature against the public key on file. Three causes:", flush=True)
        print("[google_auth]", flush=True)
        print("[google_auth] A) KEY MISMATCH (most likely):", flush=True)
        print("[google_auth]    The private key in Credentials_New.json does not match", flush=True)
        print("[google_auth]    the currently-active key registered in GCP Console.", flush=True)
        print("[google_auth]    Fix: Go to GCP Console → IAM → Service Accounts →", flush=True)
        print("[google_auth]         hrishav-majumder@stock-prices-495408.iam.gserviceaccount.com", flush=True)
        print("[google_auth]         → Keys → DELETE existing keys → ADD KEY → JSON", flush=True)
        print("[google_auth]         → Save as credentials/Credentials_New.json", flush=True)
        print("[google_auth]         → Then: export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/credentials/Credentials_New.json", flush=True)
        print("[google_auth]", flush=True)
        print("[google_auth] B) CLOCK DRIFT:", flush=True)
        print("[google_auth]    JWT iat/exp claims require system clock ≤ 5 min drift.", flush=True)
        print("[google_auth]    Fix: sudo sntp -sS time.apple.com   (macOS)", flush=True)
        print("[google_auth]         sudo ntpdate -u time.google.com  (Linux)", flush=True)
        print("[google_auth]", flush=True)
        print("[google_auth] C) APIs NOT ENABLED:", flush=True)
        print("[google_auth]    https://console.cloud.google.com/apis/library", flush=True)
        print("[google_auth]    Enable: Google Sheets API + Google Drive API", flush=True)
        print("[google_auth]    Project: stock-prices-495408", flush=True)
        print("[google_auth]", flush=True)
        print("[google_auth] Run the diagnostic for step-by-step analysis:", flush=True)
        print("[google_auth]    python3 Scripts/debug_auth.py", flush=True)

    elif "invalid_grant" in msg:
        print("[google_auth] ROOT CAUSE: invalid_grant (no JWT signature detail)", flush=True)
        print("[google_auth]    Most likely clock drift. Run:", flush=True)
        print("[google_auth]    sudo sntp -sS time.apple.com", flush=True)

    elif "permission" in msg or "forbidden" in msg or "403" in msg:
        print("[google_auth] ROOT CAUSE: Permission denied (403)", flush=True)
        print("[google_auth]    Sheet not shared with service account, or APIs disabled.", flush=True)
        print("[google_auth]    Share the sheet with:", flush=True)
        print("[google_auth]    hrishav-majumder@stock-prices-495408.iam.gserviceaccount.com", flush=True)

    elif "not found" in msg or "404" in msg:
        print("[google_auth] ROOT CAUSE: Sheet not found (404)", flush=True)
        print("[google_auth]    Check the sheet ID is correct and the sheet exists.", flush=True)

    elif "ssl" in msg or "certificate" in msg or "libressl" in msg:
        print("[google_auth] ROOT CAUSE: SSL/TLS library issue", flush=True)
        print("[google_auth]    Rebuild with Python 3.12 + OpenSSL:", flush=True)
        print("[google_auth]    ./Scripts/setup_venv.sh", flush=True)

    print(f"[google_auth] ────────────────────────────────────────────────────\n", flush=True)


def get_gspread_client(
    credentials_path: Optional[str] = None,
    scopes: Optional[List[str]] = None,
) -> Any:
    """
    Return an authorized :class:`gspread.Client` ready for Sheets operations.

    Parameters
    ----------
    credentials_path : str, optional
        Explicit path to the service-account JSON.
    scopes : list of str, optional
        OAuth scopes.  Defaults to :data:`SCOPES`.

    Returns
    -------
    gspread.Client
    """
    import gspread  # lazy import

    try:
        creds  = get_credentials(credentials_path, scopes)
        client = gspread.authorize(creds)
        print("[google_auth] gspread client authorized", flush=True)
        return client
    except Exception as exc:
        _triage_auth_error(exc)
        raise


# ── Pre-flight validation ──────────────────────────────────────────────────────

def validate_access(
    sheet_id: str,
    worksheet_name: Optional[str] = None,
    credentials_path: Optional[str] = None,
    *,
    test_write: bool = False,
) -> bool:
    """
    Run a pre-flight check against the target Google Sheet.

    Checks performed
    ----------------
    1. Credential file exists and loads without error.
    2. The spreadsheet identified by *sheet_id* is accessible.
    3. If *worksheet_name* is given, that specific tab is reachable.
    4. If *test_write* is True, attempt a no-op batch_update to confirm
       write permission (safe — sends an empty update list).

    Returns True on full success; prints a detailed report and returns False
    (or raises) on failure.
    """
    print("\n[google_auth] ── Pre-flight validation ──────────────────────────", flush=True)
    ok = True

    # 1. Auth
    try:
        client = get_gspread_client(credentials_path)
        print("[google_auth] ✓ Authentication OK", flush=True)
    except Exception as exc:
        print(f"[google_auth] ✗ Authentication FAILED: {exc}", flush=True)
        return False

    # 2. Spreadsheet access
    try:
        sh = client.open_by_key(sheet_id)
        print(f"[google_auth] ✓ Sheet access OK  → '{sh.title}'  (id={sheet_id})", flush=True)
    except Exception as exc:
        print(f"[google_auth] ✗ Sheet access FAILED for id={sheet_id}: {exc}", flush=True)
        _triage_auth_error(exc)
        return False

    # 3. Worksheet fetch
    if worksheet_name:
        try:
            ws = sh.worksheet(worksheet_name)
            row_count = ws.row_count
            print(
                f"[google_auth] ✓ Worksheet fetch OK → '{worksheet_name}' "
                f"({row_count} rows)",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[google_auth] ✗ Worksheet fetch FAILED for '{worksheet_name}': {exc}",
                flush=True,
            )
            ok = False
    else:
        ws_titles = [w.title for w in sh.worksheets()]
        print(f"[google_auth] ✓ Worksheets found : {ws_titles}", flush=True)

    # 4. Write permission
    if test_write:
        try:
            # Empty batch_update — no cells changed, but confirms write access.
            sh.batch_update({"requests": []})
            print("[google_auth] ✓ Write permission OK", flush=True)
        except Exception as exc:
            print(f"[google_auth] ✗ Write permission FAILED: {exc}", flush=True)
            ok = False

    print(
        f"[google_auth] ── Pre-flight {'PASSED' if ok else 'FAILED (see above)'} ──\n",
        flush=True,
    )
    return ok


# ── CLI convenience ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick standalone test:
        python utils/google_auth.py <sheet_id> [worksheet_name]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Validate Google Sheets access.")
    parser.add_argument("sheet_id", help="Google Sheet ID to test against.")
    parser.add_argument("worksheet", nargs="?", help="Optional worksheet name to test.")
    parser.add_argument("--credentials", default=None, help="Path to service-account JSON.")
    parser.add_argument("--write", action="store_true", help="Test write permission too.")
    args = parser.parse_args()

    success = validate_access(
        sheet_id=args.sheet_id,
        worksheet_name=args.worksheet,
        credentials_path=args.credentials,
        test_write=args.write,
    )
    sys.exit(0 if success else 1)
