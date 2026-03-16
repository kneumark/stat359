#!/usr/bin/env python3
"""Reindex output directories with numeric prefixes.

Example:
    proposal_axis_numeric_full -> 001_proposal_axis_numeric_full
"""

from __future__ import annotations

import argparse
import os

from .output_naming import reorganize_output_dirs


DEFAULT_PREFIXES = [
    "proposal_",
    "foundational",
    "instruction",
    "instruction_lora",
    "grpo",
    "before_after",
    "evaluation",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reindex output folders to numeric prefix order")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=".",
        help="Directory that contains output folders",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        default=",".join(DEFAULT_PREFIXES),
        help="Comma-separated folder prefixes to include",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=3,
        help="Zero-pad width for numeric prefixes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned renames without modifying folders",
    )

    args = parser.parse_args()

    prefixes = [prefix.strip() for prefix in args.prefixes.split(",") if prefix.strip()]
    root_dir = os.path.abspath(args.root_dir)

    rename_plan = reorganize_output_dirs(
        parent_dir=root_dir,
        prefixes=prefixes,
        width=args.width,
        dry_run=args.dry_run,
    )

    if not rename_plan:
        print("No matching folders to rename.")
        return

    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"[{mode}] {len(rename_plan)} rename(s) in: {root_dir}")
    for entry in rename_plan:
        print(f"  {entry['old_name']} -> {entry['new_name']}")


if __name__ == "__main__":
    main()
