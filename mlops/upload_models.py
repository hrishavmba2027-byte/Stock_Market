"""Push model artifacts to a GitHub Release (strict replacement).

GitHub Actions runners are ephemeral — they cannot persist a trained
``Transformer.pt`` across runs. We use a single GitHub Release tagged
``models-current`` as the durable model registry: training writes there
(monthly), daily prediction reads from there. Each upload **overwrites**
the existing assets on that release — no version history is kept on
GitHub, by design (the user asked for strict replacement).

This module shells out to the ``gh`` CLI so the same code path works:

* **Locally**, after ``gh auth login`` — bootstrap upload from your laptop.
* **In a workflow**, where ``GITHUB_TOKEN`` is auto-provided.

Usage::

    # Bootstrap from your laptop (one-time after training)
    gh auth login                                      # browser flow
    python -m mlops.upload_models

    # Or from a workflow (no extra auth needed)
    python -m mlops.upload_models --repo OWNER/REPO

If the ``models-current`` release does not yet exist it is created;
otherwise the existing assets are replaced via ``--clobber``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

DEFAULT_TAG = "models-current"
DEFAULT_TITLE = "Current model weights"
DEFAULT_NOTES = (
    "Rolling pointer to the latest trained ensemble. "
    "Replaced on every monthly retrain — no history kept."
)

ARTIFACTS = ("Dense.pt", "LSTM.pt", "Transformer.pt", "pipeline_metadata.json")
DEFAULT_LOCAL_PATHS = {
    "Dense.pt": Path("outputs/Saved_Models/Dense.pt"),
    "LSTM.pt": Path("outputs/Saved_Models/LSTM.pt"),
    "Transformer.pt": Path("outputs/Saved_Models/Transformer.pt"),
    "pipeline_metadata.json": Path("outputs/pipeline_metadata.json"),
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _resolve_local(name: str, source_dir: Path) -> Path:
    return DEFAULT_LOCAL_PATHS[name] if source_dir == Path("") else source_dir / name


def _gh_path() -> str:
    gh = shutil.which("gh")
    if not gh:
        raise RuntimeError(
            "`gh` CLI not found on PATH. Install via `brew install gh` and "
            "authenticate with `gh auth login`."
        )
    return gh


def _release_exists(gh: str, tag: str, repo: str | None) -> bool:
    cmd = [gh, "release", "view", tag]
    if repo:
        cmd += ["--repo", repo]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def _create_release(gh: str, tag: str, repo: str | None, files: List[Path]) -> None:
    cmd = [
        gh,
        "release",
        "create",
        tag,
        "--title",
        DEFAULT_TITLE,
        "--notes",
        DEFAULT_NOTES,
    ]
    if repo:
        cmd += ["--repo", repo]
    cmd += [str(p) for p in files]
    subprocess.run(cmd, check=True)


def _upload_clobber(gh: str, tag: str, repo: str | None, files: List[Path]) -> None:
    cmd = [gh, "release", "upload", tag, "--clobber"]
    if repo:
        cmd += ["--repo", repo]
    cmd += [str(p) for p in files]
    subprocess.run(cmd, check=True)


def upload(
    tag: str = DEFAULT_TAG,
    source_dir: Path = Path(""),
    only: Iterable[str] = ARTIFACTS,
    repo: str | None = None,
) -> int:
    """Push every named artifact to the rolling release tag.

    Returns the number of artifacts uploaded. Strict replacement: if the
    release already exists, the assets are overwritten via ``--clobber``.
    """
    gh = _gh_path()

    files: List[Path] = []
    for name in only:
        local = _resolve_local(name, source_dir)
        if not local.exists():
            _log(f"[upload-models] SKIP {name} (no local file at {local})")
            continue
        files.append(local)

    if not files:
        _log("[upload-models] nothing to upload — no local artifacts found")
        return 0

    if _release_exists(gh, tag, repo):
        _log(f"[upload-models] release '{tag}' exists — replacing assets")
        _upload_clobber(gh, tag, repo, files)
    else:
        _log(f"[upload-models] release '{tag}' missing — creating it")
        _create_release(gh, tag, repo, files)

    for p in files:
        _log(f"[upload-models] uploaded {p.name}")
    return len(files)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Push local model artifacts to a GitHub Release (strict replacement)."
    )
    parser.add_argument(
        "--tag",
        default=os.environ.get("MODEL_RELEASE_TAG", DEFAULT_TAG),
        help="Release tag to write to (default: 'models-current').",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="Target repo as OWNER/NAME. Defaults to the current repo when run via gh.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(""),
        help="Override directory containing the .pt files. Default uses the repo layout.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=ARTIFACTS,
        default=list(ARTIFACTS),
        help="Subset of artifacts to upload (default: all four).",
    )
    args = parser.parse_args(argv)

    written = upload(
        tag=args.tag,
        source_dir=args.source_dir,
        only=args.only,
        repo=args.repo,
    )
    if not written:
        return 1
    _log(f"[upload-models] {written} artifact(s) live at release '{args.tag}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
