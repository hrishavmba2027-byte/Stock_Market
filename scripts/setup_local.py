#!/usr/bin/env python3
"""Cross-platform bootstrap + doctor for the Stock_Market pipeline.

Makes the whole system runnable on any laptop/PC (Windows, macOS, Linux)
with nothing pre-installed except Python 3.10+:

    python scripts/setup_local.py            # create venv, install deps, doctor
    python scripts/setup_local.py --doctor   # checks only, change nothing
    python scripts/setup_local.py --with-notebook  # + jupyter/ipykernel extras

What it does, in order:
  1. Verifies the Python version.
  2. Creates ./venv if missing (using the running interpreter).
  3. Installs pinned dependencies from requirements.txt into the venv.
  4. Copies .env.example -> .env when .env is missing (then tells you which
     values still need filling in).
  5. Doctor: inside the venv it checks imports, the torch compute device
     (cuda / mps / cpu), model artifacts, metadata (including the
     circuit_limit bound block), credentials, and writable state/output dirs.
  6. Prints the exact next commands for the full workflow.

The script only ever writes inside the repository (venv/, .env) — safe to
re-run any time; every step is idempotent.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = REPO_ROOT / "venv"
MIN_PYTHON = (3, 10)

OK = "[ ok ]"
WARN = "[warn]"
FAIL = "[FAIL]"


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"       $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run([str(c) for c in cmd], check=False, **kwargs)


def step_python_version() -> bool:
    if sys.version_info < MIN_PYTHON:
        print(f"{FAIL} Python {'.'.join(map(str, MIN_PYTHON))}+ required, "
              f"found {platform.python_version()}. Install a newer Python and re-run.")
        return False
    print(f"{OK} Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    return True


def step_create_venv() -> bool:
    if venv_python().exists():
        print(f"{OK} venv already exists: {VENV_DIR}")
        return True
    print(f"       creating venv at {VENV_DIR} ...")
    venv.EnvBuilder(with_pip=True, upgrade_deps=False).create(str(VENV_DIR))
    if not venv_python().exists():
        print(f"{FAIL} venv creation did not produce {venv_python()}")
        return False
    print(f"{OK} venv created")
    return True


def step_install(with_notebook: bool) -> bool:
    req = REPO_ROOT / "requirements.txt"
    if not req.exists():
        print(f"{FAIL} requirements.txt not found at {req}")
        return False
    py = venv_python()
    if run([py, "-m", "pip", "install", "--upgrade", "pip", "-q"]).returncode != 0:
        print(f"{WARN} pip self-upgrade failed; continuing with the bundled pip")
    rc = run([py, "-m", "pip", "install", "-q", "-r", req]).returncode
    if rc != 0:
        print(f"{FAIL} dependency install failed (see pip output above)")
        return False
    print(f"{OK} requirements installed")
    if with_notebook:
        rc = run([py, "-m", "pip", "install", "-q", "jupyter", "ipykernel", "nbformat", "ipywidgets"]).returncode
        if rc != 0:
            print(f"{WARN} notebook extras failed to install")
        else:
            run([py, "-m", "ipykernel", "install", "--user",
                 "--name", "stock_market_venv", "--display-name", "Stock_Market venv"])
            print(f"{OK} notebook extras installed + 'stock_market_venv' kernel registered")
    return True


def step_dotenv() -> bool:
    env_path = REPO_ROOT / ".env"
    example = REPO_ROOT / ".env.example"
    if env_path.exists():
        print(f"{OK} .env present")
        return True
    if not example.exists():
        print(f"{FAIL} neither .env nor .env.example found")
        return False
    shutil.copyfile(example, env_path)
    print(f"{WARN} .env created from .env.example — fill in the sheet IDs, API keys "
          f"and credential paths before running live.")
    return True


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def step_doctor() -> bool:
    """Run environment checks inside the venv interpreter."""
    py = venv_python()
    if not py.exists():
        print(f"{FAIL} venv missing — run without --doctor first")
        return False

    probe = r"""
import json, os, sys
from pathlib import Path
root = Path(os.environ["SM_REPO_ROOT"])
out = {"python": sys.version.split()[0]}
try:
    import torch
    out["torch"] = torch.__version__
    if torch.cuda.is_available():
        out["device"] = "cuda (" + torch.cuda.get_device_name(0) + ")"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        out["device"] = "mps (Apple GPU)"
    else:
        out["device"] = "cpu"
except Exception as exc:
    out["torch_error"] = str(exc)
for mod in ("pandas", "numpy", "sklearn", "gspread", "plotly"):
    try:
        __import__(mod)
        out.setdefault("imports_ok", []).append(mod)
    except Exception as exc:
        out.setdefault("imports_failed", {})[mod] = str(exc)
models = {name: (root / "outputs" / "Saved_Models" / (name + ".pt")).exists()
          for name in ("Dense", "LSTM", "Transformer")}
out["model_checkpoints"] = models
meta_path = root / "outputs" / "pipeline_metadata.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    out["metadata"] = {
        "seq_len": meta.get("seq_len"),
        "feature_count": meta.get("feature_count"),
        "circuit_limit_bound": bool(meta.get("circuit_limit")),
    }
else:
    out["metadata"] = "missing"
for rel in ("state", "outputs", "logs"):
    p = root / rel
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe_file = p / ".write_probe"
        probe_file.write_text("ok")
        probe_file.unlink()
        out.setdefault("writable", []).append(rel)
    except Exception as exc:
        out.setdefault("not_writable", {})[rel] = str(exc)
print(json.dumps(out))
"""
    env = dict(os.environ, SM_REPO_ROOT=str(REPO_ROOT))
    proc = subprocess.run([str(py), "-c", probe], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"{FAIL} doctor probe crashed:\n{proc.stderr[-800:]}")
        return False
    report = json.loads(proc.stdout.strip().splitlines()[-1])

    healthy = True
    print(f"{OK} venv python {report.get('python')} | torch {report.get('torch', 'MISSING')} "
          f"| device: {report.get('device', 'n/a')}")
    if report.get("torch_error") or report.get("imports_failed"):
        healthy = False
        print(f"{FAIL} import problems: {report.get('torch_error', '')} {report.get('imports_failed', '')}")
    missing_models = [n for n, ok in report["model_checkpoints"].items() if not ok]
    if missing_models:
        print(f"{WARN} missing checkpoints: {missing_models} — train via the notebook or "
              f"download the 'models-current' release before forecasting.")
    else:
        print(f"{OK} model checkpoints present (Dense/LSTM/Transformer)")
    meta = report.get("metadata")
    if isinstance(meta, dict):
        bound = "with circuit-limit bound" if meta.get("circuit_limit_bound") else "WITHOUT circuit-limit bound"
        print(f"{OK} metadata: seq_len={meta.get('seq_len')} features={meta.get('feature_count')} ({bound})")
    else:
        print(f"{WARN} pipeline_metadata.json missing — run training first")
    if report.get("not_writable"):
        healthy = False
        print(f"{FAIL} not writable: {report['not_writable']}")

    env_values = _parse_env_file(REPO_ROOT / ".env")
    required_env = ["TRAIN_END", "TEST_END", "BACK_TEST_START", "BACK_TEST_END"]
    live_env = ["OPERATIONAL_SHEET_ID", "HISTORICAL_TRAINING_SHEET_ID"]
    missing_req = [k for k in required_env if not env_values.get(k)]
    missing_live = [k for k in live_env if not env_values.get(k) or "your_" in env_values.get(k, "")]
    if missing_req:
        healthy = False
        print(f"{FAIL} .env missing required keys: {missing_req}")
    else:
        print(f"{OK} .env split dates present")
    if missing_live:
        print(f"{WARN} .env live-mode keys unset/placeholder: {missing_live} "
              f"(fine for local dry-runs; required for --live and fine-tuning)")

    cred_candidates = [
        env_values.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
        str(REPO_ROOT / "credentials" / "Credentials_New.json"),
    ]
    if any(c and Path(c).expanduser().exists() for c in cred_candidates):
        print(f"{OK} Google service-account credentials found")
    else:
        print(f"{WARN} no Google credentials file found — Sheets stages and fine-tuning "
              f"will fail until credentials/ is populated")
    return healthy


def print_next_steps() -> None:
    activate = ("venv\\Scripts\\activate" if platform.system() == "Windows"
                else "source venv/bin/activate")
    print(
        "\nNext steps\n"
        "----------\n"
        f"  1. {activate}\n"
        "  2. python run_full_workflow.py            # dry-run rehearsal (16 stages)\n"
        "  3. python run_full_workflow.py --live     # real run: data -> forecasts -> sheets\n"
        "                                            # stage 15 auto fine-tunes monthly (--if-due)\n"
        "  4. python -m ingestion.collect_all        # news/reddit/X + FinBERT sentiment -> Firestore\n"
        "  5. python -m features.trade_suggestions   # LLM decisions -> Firestore trade_suggestions\n"
        "\n"
        "Manual fine-tune controls:\n"
        "  python monthly_finetune.py --check-only   # is a retrain due? (read-only)\n"
        "  python monthly_finetune.py --if-due       # fine-tune only if due\n"
        "  python monthly_finetune.py                # force incremental fine-tune now\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap + doctor for the Stock_Market pipeline.")
    parser.add_argument("--doctor", action="store_true", help="Run checks only; change nothing.")
    parser.add_argument("--with-notebook", action="store_true",
                        help="Also install jupyter/ipykernel/nbformat/ipywidgets and register the kernel.")
    args = parser.parse_args()

    print(f"Stock_Market local setup — repo: {REPO_ROOT}\n")
    if not step_python_version():
        return 1
    if not args.doctor:
        if not step_create_venv():
            return 1
        if not step_install(args.with_notebook):
            return 1
        if not step_dotenv():
            return 1
    healthy = step_doctor()
    print_next_steps()
    if not healthy:
        print("Doctor found blocking problems above — fix them before running the workflow.")
        return 1
    print("Setup complete — environment is healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
