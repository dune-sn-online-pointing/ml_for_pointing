#!/usr/bin/env python3
"""Merge NPZ files that share a common basename but differ by a suffix token.

Typical use case: directories that contain many files like
```
<basename>_planeU.npz
<basename>_planeV.npz
<basename>_planeX.npz
```
This utility consolidates the groups into a single file per basename, keeping
all arrays stacked together and storing metadata arrays as larger batches.

Original files are removed after the merged file is safely written.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class MergeStats:
    processed_groups: int = 0
    skipped_existing: int = 0
    missing_arrays: int = 0


def find_groups(directory: Path, token: str, extension: str) -> Dict[str, List[Path]]:
    """Group files by removing the suffix introduced by *token*.

    For example, with token="_plane" and filename
    "event_bg_matched_planeX.npz", the group key is
    "event_bg_matched".
    """

    groups: Dict[str, List[Path]] = {}

    with os.scandir(directory) as iterator:
        for entry in iterator:
            if not entry.is_file():
                continue
            name = entry.name
            if not name.endswith(extension):
                continue
            idx = name.rfind(token)
            if idx == -1:
                continue
            base = name[:idx]
            key = base
            groups.setdefault(key, []).append(directory / name)

    return groups


def merge_group(paths: List[Path], output_path: Path) -> bool:
    """Merge a list of NPZ files into *output_path*.

    Returns True on success, False if required arrays are missing.
    """

    if len(paths) < 2:
        return False

    images_list: List[np.ndarray] = []
    metadata_list: List[np.ndarray] = []

    sorted_paths = sorted(paths)
    for path in sorted_paths:
        with np.load(path, allow_pickle=True) as data:
            if "images" not in data or "metadata" not in data:
                return False
            images = data["images"]
            metadata = data["metadata"]
            images_list.append(images)
            metadata_list.append(metadata)

    merged_images = np.concatenate(images_list, axis=0)
    merged_metadata = np.concatenate(metadata_list, axis=0)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with tmp_path.open("wb") as buffer:
            np.savez_compressed(buffer, images=merged_images, metadata=merged_metadata)
        os.replace(tmp_path, output_path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        return False

    remove_sources(sorted_paths)
    return True


def remove_sources(paths: List[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def merge_directory(
    directory: Path,
    token: str,
    extension: str,
    max_groups: int | None,
    dry_run: bool,
    min_members: int,
) -> MergeStats:
    groups = find_groups(directory, token, extension)
    stats = MergeStats()

    if not groups:
        return stats

    # Deterministic order by basename
    for key in sorted(groups.keys()):
        paths = groups[key]
        output_path = directory / f"{key}{extension}"

        if output_path.exists():
            stats.skipped_existing += 1
            continue

        if len(paths) < min_members:
            stats.missing_arrays += 1
            continue

        stats.processed_groups += 1

        if dry_run:
            if max_groups is not None and stats.processed_groups >= max_groups:
                break
            continue

        success = merge_group(paths, output_path)
        if not success:
            stats.missing_arrays += 1
            # Leave source files intact for further inspection
            output_path.unlink(missing_ok=True)
        if max_groups is not None and stats.processed_groups >= max_groups:
            break

    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Directory containing NPZ files to merge")
    parser.add_argument(
        "--token",
        default="_plane",
        help="Suffix token that marks the split between basename and suffix (default: _plane)",
    )
    parser.add_argument(
        "--extension",
        default=".npz",
        help="File extension to process (default: .npz)",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Maximum number of groups to process in this run",
    )
    parser.add_argument(
        "--min-members",
        type=int,
        default=2,
        help="Minimum number of files required in a group to attempt merge",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List groups without writing or deleting files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    directory = Path(args.directory).expanduser()

    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        return 1

    stats = merge_directory(
        directory=directory,
        token=args.token,
        extension=args.extension,
        max_groups=args.max_groups,
        dry_run=args.dry_run,
        min_members=args.min_members,
    )

    print(
        f"Processed groups: {stats.processed_groups}, "
        f"skipped existing: {stats.skipped_existing}, "
        f"missing arrays: {stats.missing_arrays}"
    )

    if args.max_groups is not None:
        print(f"Stopped after reaching max_groups={args.max_groups}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
