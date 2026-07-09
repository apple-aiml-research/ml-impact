#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import argparse
from pathlib import Path
from typing import Optional

from hydra import initialize, compose

from src.config import CODE2LANG
from src.inference.inference_pipeline import process_one_file


def get_latest_file(scenario_dir: Path, filename: str) -> Path:
    """Return the most recent file with name `filename` in `scenario_dir`."""
    timestamp_dirs = [p for p in scenario_dir.iterdir() if p.is_dir()]
    if not timestamp_dirs:
        return None
    latest = max(timestamp_dirs, key=lambda p: p.stat().st_mtime)
    data_file = latest / filename
    return data_file if data_file.exists() else None


def collect_files(base_dir: Path, scenario: Optional[str] = None):
    """
    Collects the latest data.json file from each template/scenario/timestamp directory.
    Optionally filters by scenario.
    """
    files = []
    # Detect if base_dir is already pointing to a language folder
    if base_dir.name in CODE2LANG:
        lang_dirs = [base_dir]
    else:
        lang_dirs = [d for d in base_dir.glob("*") if d.is_dir()]

    for lang_dir in lang_dirs:
        template_dirs = [t for t in lang_dir.glob("*") if t.is_dir()]
        for template_dir in template_dirs:
            scenario_dirs = [s for s in template_dir.glob("*") if s.is_dir()]
            for s_dir in scenario_dirs:
                if scenario and s_dir.name != scenario:
                    continue
                latest_file = get_latest_file(s_dir, "sampled_data.json")
                if latest_file:
                    files.append(latest_file)
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base_dir", default="data/generated", help="where your template folders live"
    )
    group = p.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--file", type=str, help="Run inference on a single file (JSON)."
    )
    group.add_argument(
        "--lang_code",
        choices=list(CODE2LANG.keys()),
        help="Only process files for this language code.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all files across all languages and scenarios.",
    )
    p.add_argument(
        "--scenario",
        choices=["generation", "judge"],
        required=False,
        help="If set, only process files under this scenario subfolder; otherwise process all.",
    )
    p.add_argument(
        "--config_name", default="config", help="Hydra config name (without .yaml)"
    )
    p.add_argument(
        "hydra_overrides",
        nargs="*",
        help="any extra hydra overrides: model=..., system_instructions.value=...",
    )
    p.add_argument("--limit", type=int, help="Set a limit (must be an integer)")

    args = p.parse_args()

    if args.file:
        to_run = [Path(args.file)]
        if not args.scenario:
            if "generation" in args.file:
                args.scenario = "generation"
            elif "judge" in args.file:
                args.scenario = "judge"
            else:
                raise ValueError("Could not identify the scenario.")
    else:
        base = Path(args.base_dir)
        if args.lang_code:
            base = base / args.lang_code
        if not args.scenario:
            p.error("--scenario is required.")
        to_run = collect_files(base, args.scenario)

    if not to_run:
        print("No files to process.")
        return

    for fpath in to_run:
        print("→ processing", fpath)
        with initialize(config_path="../src/inference/config", version_base=None):
            # Compose config, override filename + any extra flags
            overrides = [
                f"filename={fpath}",
                f"scenario={args.scenario}",
            ] + args.hydra_overrides
            cfg = compose(config_name=args.config_name, overrides=overrides)
            process_one_file(cfg, args.limit)


if __name__ == "__main__":
    main()
