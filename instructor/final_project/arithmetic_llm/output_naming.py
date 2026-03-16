"""Utilities for consistent numeric output directory naming."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

_INDEXED_PATTERN = re.compile(r"^(\d{3,})_(.+)$")


def _parse_indexed_name(name: str) -> Optional[Tuple[int, str]]:
    match = _INDEXED_PATTERN.match(name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def get_next_index(parent_dir: str) -> int:
    os.makedirs(parent_dir, exist_ok=True)
    max_index = 0
    for entry in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, entry)
        if not os.path.isdir(full_path):
            continue
        parsed = _parse_indexed_name(entry)
        if parsed is None:
            continue
        max_index = max(max_index, parsed[0])
    return max_index + 1


def create_numbered_output_dir(parent_dir: str, base_name: str, width: int = 3) -> str:
    os.makedirs(parent_dir, exist_ok=True)
    next_index = get_next_index(parent_dir)

    while True:
        candidate_name = f"{next_index:0{width}d}_{base_name}"
        candidate_path = os.path.join(parent_dir, candidate_name)
        if not os.path.exists(candidate_path):
            os.makedirs(candidate_path, exist_ok=False)
            return candidate_path
        next_index += 1


def _matches_prefix(logical_name: str, prefixes: Sequence[str]) -> bool:
    return any(logical_name.startswith(prefix) for prefix in prefixes)


def reorganize_output_dirs(
    parent_dir: str,
    prefixes: Sequence[str],
    width: int = 3,
    dry_run: bool = True,
) -> List[Dict[str, str]]:
    """Rename matching output directories to contiguous numeric order.

    Ordering is by modification time ascending, so the latest run gets the largest index.
    """
    os.makedirs(parent_dir, exist_ok=True)

    candidates: List[Tuple[str, str, float]] = []
    for entry in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, entry)
        if not os.path.isdir(full_path):
            continue

        parsed = _parse_indexed_name(entry)
        logical_name = parsed[1] if parsed is not None else entry
        if not _matches_prefix(logical_name, prefixes):
            continue

        modified_time = os.path.getmtime(full_path)
        candidates.append((entry, logical_name, modified_time))

    candidates.sort(key=lambda item: item[2])

    plan: List[Tuple[str, str, str]] = []
    for idx, (old_name, logical_name, _) in enumerate(candidates, start=1):
        new_name = f"{idx:0{width}d}_{logical_name}"
        if old_name != new_name:
            plan.append((old_name, new_name, logical_name))

    if dry_run or not plan:
        return [
            {
                "old_name": old_name,
                "new_name": new_name,
                "logical_name": logical_name,
            }
            for old_name, new_name, logical_name in plan
        ]

    temporary_pairs: List[Tuple[str, str]] = []
    for old_name, _, _ in plan:
        old_path = os.path.join(parent_dir, old_name)
        temp_name = f"__tmp_reindex__{old_name}"
        temp_path = os.path.join(parent_dir, temp_name)
        os.replace(old_path, temp_path)
        temporary_pairs.append((temp_name, old_name))

    temp_to_new = {
        f"__tmp_reindex__{old_name}": new_name
        for old_name, new_name, _ in plan
    }

    for temp_name, _ in temporary_pairs:
        temp_path = os.path.join(parent_dir, temp_name)
        new_name = temp_to_new[temp_name]
        new_path = os.path.join(parent_dir, new_name)
        os.replace(temp_path, new_path)

    return [
        {
            "old_name": old_name,
            "new_name": new_name,
            "logical_name": logical_name,
        }
        for old_name, new_name, logical_name in plan
    ]
