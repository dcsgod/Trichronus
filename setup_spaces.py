"""
setup_spaces.py — TriChronos-0.1B
Creates and deploys both HF Spaces (training + eval) for iravikr/trichronos-*.

Pattern learned from ICML CAOS project:
  - Use sdk: gradio (not docker) with `hardware: a100-large` in the README YAML.
    This specifies a PAID hardware tier at create time, bypassing the free-CPU
    PRO gate — no HF Pro subscription needed.
  - Deploy via git push (not upload_folder API), matching huggingface-cli workflow.
  - Use a HF Dataset repo as a persistent checkpoint bucket (mounted at /data).

Usage
-----
  HF_TOKEN=hf_... python setup_spaces.py
  python setup_spaces.py --dry-run    # print actions, no API calls

What this script does
---------------------
  1. Creates iravikr/trichronos-checkpoints  (Dataset repo — checkpoint bucket)
  2. Creates iravikr/trichronos-train        (Gradio Space, hardware=a100-large)
  3. Creates iravikr/trichronos-eval         (Gradio Space, hardware=cpu-upgrade)
  4. Uploads all source files to both Spaces via upload_folder
  5. Sets HF_TOKEN + SPACE_ID as Space secrets programmatically
  6. Prints monitoring URLs
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USERNAME = "iravikr"
TRAIN_SPACE_ID = f"{USERNAME}/trichronos-train"
EVAL_SPACE_ID = f"{USERNAME}/trichronos-eval"
CHECKPOINT_BUCKET_ID = f"{USERNAME}/trichronos-checkpoints"   # Dataset repo = bucket

PROJECT_ROOT = Path(__file__).parent.resolve()
SPACES_DIR = PROJECT_ROOT / "spaces"
TRAIN_SPACE_DIR = SPACES_DIR / "train"
EVAL_SPACE_DIR = SPACES_DIR / "eval"

SHARED_SOURCE_FILES = ["bitlinear.py", "data_pipeline.py", "model.py", "evaluate.py"]
TRAIN_ONLY_FILES    = ["train.py"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hdr(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


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
            shutil.copytree(
                item, dest, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

    for fname in extra_files:
        src = PROJECT_ROOT / fname
        if src.exists():
            shutil.copy2(src, staging / fname)
        else:
            print(f"  [WARNING] Not found: {src}", file=sys.stderr)

    return staging


def _list_staging(staging: Path):
    print(f"  Staged files ({staging.name}):")
    for f in sorted(staging.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(staging)}  ({f.stat().st_size/1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Bucket (Dataset repo for persistent checkpoints)
# ---------------------------------------------------------------------------

def _create_bucket(api, dry_run: bool):
    _hdr("Checkpoint bucket")
    print(f"  Repo : {CHECKPOINT_BUCKET_ID}")
    print(f"  Type : dataset  (mounted at /data inside the Space)")

    if dry_run:
        print("  [DRY RUN] Would create dataset repo for checkpoints")
        return

    try:
        api.create_repo(
            repo_id=CHECKPOINT_BUCKET_ID,
            repo_type="dataset",
            exist_ok=True,
            private=False,
        )
        print(f"  Created / verified: https://huggingface.co/datasets/{CHECKPOINT_BUCKET_ID}")
    except Exception as exc:
        print(f"  [ERROR] Could not create bucket: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Space creation + upload
# ---------------------------------------------------------------------------

def _create_and_upload(
    api,
    space_id: str,
    staging: Path,
    token: str,
    space_variables: list[dict] | None = None,
    space_secrets: list[dict] | None = None,
    dry_run: bool = False,
):
    print(f"\n  Space: {space_id}")
    print(f"  URL  : https://huggingface.co/spaces/{space_id}")

    if dry_run:
        print(f"  [DRY RUN] Would create Gradio Space (hardware in README YAML)")
        _list_staging(staging)
        return

    # --- Create Space ---
    # Key insight (from ICML pattern):
    # The hardware tier is declared in the README.md YAML frontmatter (`hardware: a100-large`).
    # When create_repo reads the uploaded README, HF automatically sets the hardware.
    # This bypasses the free-CPU PRO gate because we're committing to a paid tier.
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False,
            space_secrets=space_secrets or [],
            space_variables=space_variables or [],
        )
        print(f"  Created / verified space.")
    except Exception as exc:
        print(f"  [ERROR] create_repo failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Upload files ---
    print(f"  Uploading ...")
    _list_staging(staging)
    try:
        api.upload_folder(
            folder_path=str(staging),
            repo_id=space_id,
            repo_type="space",
            commit_message="TriChronos-0.1B setup",
            ignore_patterns=["__pycache__", "*.pyc", ".DS_Store"],
        )
        print(f"  Upload complete.")
    except Exception as exc:
        print(f"  [ERROR] Upload failed: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Post-setup summary
# ---------------------------------------------------------------------------

def _print_summary(dry_run: bool):
    prefix = "[DRY RUN] " if dry_run else ""
    _hdr(f"{prefix}Done")

    lines = [
        "",
        "CHECKPOINT BUCKET",
        f"  https://huggingface.co/datasets/{CHECKPOINT_BUCKET_ID}",
        "  Mounted at /data inside the training Space.",
        "  Checkpoints survive Space restarts automatically.",
        "",
        "TRAINING SPACE",
        f"  https://huggingface.co/spaces/{TRAIN_SPACE_ID}",
        "  Hardware: A100-80GB (a100-large) — set via README YAML, no Pro needed.",
        "  HF_TOKEN + SPACE_ID secrets set automatically by this script.",
        "  Training starts when you click Start in the Gradio UI.",
        "",
        "EVAL SPACE",
        f"  https://huggingface.co/spaces/{EVAL_SPACE_ID}",
        "  Hardware: cpu-upgrade (paid, ~$0.05/hr) — no Pro needed.",
        "  Upload a checkpoint -> MASE scores stream per Monash dataset.",
        "",
        "COST",
        "  A100: ~$2.50/hr  |  Hard cutoff 7h10m  |  Max ~$17.92",
        "  cpu-upgrade: ~$0.05/hr (eval space, idle-sleeps after 1h)",
        "",
        "AFTER TRAINING",
        "  python publish.py --repo-id iravikr/trichronos-0.1b",
        "",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Set up TriChronos HF Spaces")
    p.add_argument("--token", default="", help="HF write token (or set HF_TOKEN env var)")
    p.add_argument("--dry-run", action="store_true", help="No API calls — print what would happen")
    return p.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN", "")

    if not token and not args.dry_run:
        print("ERROR: HF write token required. Set HF_TOKEN or pass --token hf_...",
              file=sys.stderr)
        sys.exit(1)

    _hdr("TriChronos-0.1B HF Spaces Setup")
    print(f"  Pattern    : Gradio SDK + hardware in README YAML (ICML approach)")
    print(f"  Train Space: {TRAIN_SPACE_ID}  (a100-large)")
    print(f"  Eval Space : {EVAL_SPACE_ID}  (cpu-upgrade)")
    print(f"  Bucket     : {CHECKPOINT_BUCKET_ID}  (dataset repo)")
    print(f"  Dry run    : {args.dry_run}")

    for d in [TRAIN_SPACE_DIR, EVAL_SPACE_DIR]:
        if not d.is_dir():
            print(f"ERROR: Missing directory: {d}", file=sys.stderr)
            sys.exit(1)

    api = None
    if not args.dry_run:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        try:
            user = api.whoami()
            print(f"  Authenticated: {user['name']}")
        except Exception as exc:
            print(f"ERROR: Auth failed: {exc}", file=sys.stderr)
            sys.exit(1)

    # 1. Checkpoint bucket
    _create_bucket(api, dry_run=args.dry_run)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 2. Training Space
        _hdr("Training Space")
        train_staging = _assemble_staging(TRAIN_SPACE_DIR, SHARED_SOURCE_FILES + TRAIN_ONLY_FILES, tmpdir)
        _create_and_upload(
            api=api,
            space_id=TRAIN_SPACE_ID,
            staging=train_staging,
            token=token,
            space_secrets=[{"key": "HF_TOKEN",  "value": token}],
            space_variables=[
                {"key": "SPACE_ID",             "value": TRAIN_SPACE_ID},
                {"key": "CHECKPOINT_BUCKET_ID", "value": CHECKPOINT_BUCKET_ID},
            ],
            dry_run=args.dry_run,
        )

        # 3. Eval Space
        _hdr("Eval Space")
        eval_staging = _assemble_staging(EVAL_SPACE_DIR, SHARED_SOURCE_FILES, tmpdir)
        _create_and_upload(
            api=api,
            space_id=EVAL_SPACE_ID,
            staging=eval_staging,
            token=token,
            space_secrets=[{"key": "HF_TOKEN",  "value": token}],
            space_variables=[
                {"key": "CHECKPOINT_BUCKET_ID", "value": CHECKPOINT_BUCKET_ID},
            ],
            dry_run=args.dry_run,
        )

    _print_summary(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
