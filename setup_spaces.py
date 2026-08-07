"""
setup_spaces.py — TriChronos-0.1B
Creates and deploys both HF Spaces (training + eval) for iravikr/trichronos-*.

Uses `hf repos create` CLI with --flavor (paid hardware) to bypass the
free-CPU PRO gate — no HF Pro subscription needed.
Also creates a Dataset bucket repo for persistent checkpoint storage.

Usage
-----
  # Token from env (already set):
  python setup_spaces.py

  # Or pass explicitly:
  python setup_spaces.py --token hf_...

  # Dry-run (no API calls):
  python setup_spaces.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USERNAME             = "iravikr"
TRAIN_SPACE_ID       = f"{USERNAME}/trichronos-train"
EVAL_SPACE_ID        = f"{USERNAME}/trichronos-eval"
CHECKPOINT_BUCKET_ID = f"{USERNAME}/trichronos-checkpoints"

PROJECT_ROOT    = Path(__file__).parent.resolve()
SPACES_DIR      = PROJECT_ROOT / "spaces"
TRAIN_SPACE_DIR = SPACES_DIR / "train"
EVAL_SPACE_DIR  = SPACES_DIR / "eval"

SHARED_SOURCE_FILES = ["bitlinear.py", "data_pipeline.py", "model.py", "evaluate.py"]
TRAIN_ONLY_FILES    = ["train.py"]

# hf CLI binary (resolved from PATH or venv)
HF_CLI = shutil.which("hf") or "hf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hdr(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def _run(cmd: list[str], dry_run: bool, token: str = "") -> bool:
    """Run a shell command, injecting HF_TOKEN into env. Returns True on success."""
    env = {**os.environ}
    if token:
        env["HF_TOKEN"] = token

    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        print("  [DRY RUN] skipped")
        return True

    result = subprocess.run(cmd, env=env, capture_output=False, text=True)
    if result.returncode != 0:
        return False
    return True


def _assemble_staging(space_dir: Path, extra_files: list[str], tmpdir: Path) -> Path:
    staging = tmpdir / space_dir.name
    staging.mkdir(parents=True, exist_ok=True)

    for item in space_dir.iterdir():
        if item.name in ("__pycache__", ".git"):
            continue
        dest = staging / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    for fname in extra_files:
        src = PROJECT_ROOT / fname
        if src.exists():
            shutil.copy2(src, staging / fname)
        else:
            print(f"  [WARNING] Not found: {src}", file=sys.stderr)

    return staging


def _list_staging(staging: Path):
    print(f"  Files to upload ({staging.name}):")
    for f in sorted(staging.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(staging)}  ({f.stat().st_size / 1024:.1f} KB)")


def _upload(staging: Path, repo_id: str, repo_type: str, token: str, dry_run: bool):
    """Upload staging dir via huggingface_hub.upload_folder."""
    _list_staging(staging)
    if dry_run:
        print("  [DRY RUN] upload skipped")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="TriChronos-0.1B setup",
        ignore_patterns=["__pycache__", "*.pyc", ".DS_Store", "Dockerfile"],
    )
    print("  Upload complete.")


# ---------------------------------------------------------------------------
# Step 1: Checkpoint bucket
# ---------------------------------------------------------------------------

def create_bucket(token: str, dry_run: bool):
    _hdr("Step 1: Checkpoint bucket (Dataset repo)")
    print(f"  Repo : {CHECKPOINT_BUCKET_ID}")
    print(f"  Type : dataset  (will be mounted at /data in the training Space)")

    ok = _run([
        HF_CLI, "repos", "create", CHECKPOINT_BUCKET_ID,
        "--type", "dataset",
        "--exist-ok",
    ], dry_run=dry_run, token=token)

    if not ok:
        print("  [ERROR] Failed to create bucket repo.", file=sys.stderr)
        sys.exit(1)
    print(f"  https://huggingface.co/datasets/{CHECKPOINT_BUCKET_ID}")


# ---------------------------------------------------------------------------
# Step 2+3: Create Spaces with paid hardware via hf CLI
# ---------------------------------------------------------------------------

def create_space(
    space_id: str,
    flavor: str,
    secrets: list[tuple[str, str]],
    env_vars: list[tuple[str, str]],
    volume: str | None,
    token: str,
    dry_run: bool,
):
    """
    Create a Gradio Space using `hf repos create` with paid --flavor.
    Paid hardware tiers bypass the free-CPU PRO gate.
    """
    cmd = [
        HF_CLI, "repos", "create", space_id,
        "--type", "space",
        "--sdk",  "gradio",
        "--flavor", flavor,
        "--exist-ok",
    ]
    for key, val in secrets:
        cmd += ["--secrets", f"{key}={val}"]
    for key, val in env_vars:
        cmd += ["--env", f"{key}={val}"]
    if volume:
        cmd += ["--volume", volume]

    ok = _run(cmd, dry_run=dry_run, token=token)
    if not ok:
        print(f"  [ERROR] Failed to create space {space_id}", file=sys.stderr)
        sys.exit(1)
    print(f"  https://huggingface.co/spaces/{space_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Set up TriChronos HF Spaces")
    p.add_argument("--token", default="", help="HF write token (or set HF_TOKEN env var)")
    p.add_argument("--dry-run", action="store_true", help="Print actions, no API calls")
    return p.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN", "")

    if not token and not args.dry_run:
        print("ERROR: HF_TOKEN required. Set env var or pass --token hf_...", file=sys.stderr)
        sys.exit(1)

    _hdr("TriChronos-0.1B HF Spaces Setup")
    print(f"  CLI        : {HF_CLI}")
    print(f"  Train Space: {TRAIN_SPACE_ID}  (flavor=a100-large, $2.50/hr)")
    print(f"  Eval Space : {EVAL_SPACE_ID}  (flavor=cpu-upgrade, $0.03/hr)")
    print(f"  Bucket     : {CHECKPOINT_BUCKET_ID}  (dataset repo, mounted at /data)")
    print(f"  Dry run    : {args.dry_run}")

    if not args.dry_run:
        # Quick auth check
        result = subprocess.run(
            [HF_CLI, "auth", "whoami"],
            env={**os.environ, "HF_TOKEN": token},
            capture_output=True, text=True,
        )
        name = result.stdout.strip() or "unknown"
        print(f"  Authenticated as: {name}")

    for d in [TRAIN_SPACE_DIR, EVAL_SPACE_DIR]:
        if not d.is_dir():
            print(f"ERROR: Missing: {d}", file=sys.stderr)
            sys.exit(1)

    # ---- 1. Bucket ----
    create_bucket(token=token, dry_run=args.dry_run)

    # ---- 2. Training Space ----
    _hdr("Step 2: Training Space")
    create_space(
        space_id=TRAIN_SPACE_ID,
        flavor="a100-large",
        secrets=[("HF_TOKEN", token)],
        env_vars=[
            ("TRICHRONOS_SPACE_ID",  TRAIN_SPACE_ID),
            ("CHECKPOINT_BUCKET_ID", CHECKPOINT_BUCKET_ID),
        ],
        # Mount the checkpoint bucket read+write at /data
        volume=f"hf://datasets/{CHECKPOINT_BUCKET_ID}:/data",
        token=token,
        dry_run=args.dry_run,
    )

    # ---- 3. Eval Space ----
    _hdr("Step 3: Eval Space")
    create_space(
        space_id=EVAL_SPACE_ID,
        flavor="cpu-upgrade",
        secrets=[("HF_TOKEN", token)],
        env_vars=[("CHECKPOINT_BUCKET_ID", CHECKPOINT_BUCKET_ID)],
        volume=f"hf://datasets/{CHECKPOINT_BUCKET_ID}:/data:ro",  # read-only
        token=token,
        dry_run=args.dry_run,
    )

    # ---- 4. Upload source files ----
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        _hdr("Step 4: Upload training Space files")
        train_staging = _assemble_staging(
            TRAIN_SPACE_DIR, SHARED_SOURCE_FILES + TRAIN_ONLY_FILES, tmpdir
        )
        _upload(train_staging, TRAIN_SPACE_ID, "space", token, args.dry_run)

        _hdr("Step 5: Upload eval Space files")
        eval_staging = _assemble_staging(EVAL_SPACE_DIR, SHARED_SOURCE_FILES, tmpdir)
        _upload(eval_staging, EVAL_SPACE_ID, "space", token, args.dry_run)

    # ---- Summary ----
    _hdr("Done" + (" [DRY RUN]" if args.dry_run else ""))
    lines = [
        "",
        "CHECKPOINT BUCKET",
        f"  https://huggingface.co/datasets/{CHECKPOINT_BUCKET_ID}",
        "  Mounted read+write at /data in training Space.",
        "  Mounted read-only  at /data in eval Space.",
        "",
        "TRAINING SPACE",
        f"  https://huggingface.co/spaces/{TRAIN_SPACE_ID}",
        "  Hardware : A100-80GB ($2.50/hr) — created with paid flavor, no Pro needed.",
        "  Secrets  : HF_TOKEN, SPACE_ID set automatically.",
        "  Volume   : checkpoint bucket mounted at /data.",
        "  Action   : click Start in the Gradio UI to begin training.",
        "",
        "EVAL SPACE",
        f"  https://huggingface.co/spaces/{EVAL_SPACE_ID}",
        "  Hardware : cpu-upgrade ($0.03/hr)",
        "  Action   : upload checkpoint -> MASE scores stream per Monash dataset.",
        "",
        "AFTER TRAINING",
        "  python publish.py --repo-id iravikr/trichronos-0.1b",
        "",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
