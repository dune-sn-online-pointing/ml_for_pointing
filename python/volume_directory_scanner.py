#!/usr/bin/env python3
"""Utility to iterate over large volume-image directories without shell globbing.

This script walks a directory tree, loads NPZ files lazily, and extracts
metadata in manageable chunks so we can inspect contents without hitting the
"Argument list too long" shell error. It understands both the standard cluster
metadata arrays and the volume metadata dictionaries produced by
`scripts/create_volumes.py`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np

# Allow re-use of the existing metadata parser for cluster-style datasets.
try:
    import data_loader
except ImportError:  # pragma: no cover - executed when run from repo root
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    import data_loader  # type: ignore


@dataclass
class ScanResult:
    files_scanned: int
    entries_seen: int
    interaction_counts: Counter
    es_flags: Counter
    main_track_flags: Counter
    plane_counts: Counter
    marley_flags: Counter
    samples: List[Dict]


def iter_npz_files(root: Path, pattern: str, recursive: bool) -> Iterator[Path]:
    """Yield NPZ files under *root* lazily."""

    if recursive:
        iterator: Iterable[Path] = root.rglob(pattern)
    else:
        iterator = root.glob(pattern)

    for candidate in iterator:
        if candidate.is_file():
            yield candidate


def unpack_metadata_array(metadata_array: np.ndarray) -> List[Dict]:
    """Normalize metadata arrays from both cluster and volume datasets."""

    if metadata_array.size == 0:
        return []

    if metadata_array.dtype == object:
        normalized: List[Dict] = []
        for item in metadata_array:
            value = item
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            if isinstance(value, dict):
                normalized.append(value)
            else:
                raise TypeError(
                    "Unsupported metadata object. Expected dict entries for volume datasets."
                )
        return normalized

    # Cluster datasets store flat float arrays; reuse the project helper.
    parsed: List[Dict] = []
    for row in metadata_array:
        parsed.append(data_loader.parse_metadata(row))
    return parsed


def load_metadata(npz_path: Path) -> List[Dict]:
    """Load metadata from a single NPZ file."""

    with np.load(npz_path, allow_pickle=True) as data:
        if "metadata" not in data:
            return []
        metadata_array = data["metadata"]
    return unpack_metadata_array(metadata_array)


def scan_directory(
    root: Path,
    pattern: str,
    recursive: bool,
    limit: int | None,
    skip: int,
    sample_count: int,
    verbose: bool,
) -> ScanResult:
    """Walk a directory, accumulating lightweight metadata statistics."""

    interaction_counts: Counter = Counter()
    es_flags: Counter = Counter()
    main_track_flags: Counter = Counter()
    marley_flags: Counter = Counter()
    plane_counts: Counter = Counter()
    samples: List[Dict] = []

    files_scanned = 0
    entries_seen = 0

    iterator = iter_npz_files(root, pattern, recursive)

    for idx, npz_path in enumerate(iterator):
        if idx < skip:
            continue
        if limit is not None and files_scanned >= limit:
            break

        try:
            metadata_entries = load_metadata(npz_path)
        except Exception as exc:  # pragma: no cover - informational
            if verbose:
                print(f"Failed to load {npz_path}: {exc}")
            continue

        files_scanned += 1

        if verbose and files_scanned % 200 == 0:
            print(f"Processed {files_scanned} files ...")

        for entry in metadata_entries:
            entries_seen += 1
            interaction = entry.get("interaction_type")
            if interaction is not None:
                interaction_counts[interaction] += 1

            is_es = entry.get("is_es_interaction")
            if is_es is not None:
                es_flags[bool(is_es)] += 1

            is_main = entry.get("is_main_track")
            if is_main is not None:
                main_track_flags[bool(is_main)] += 1

            is_marley = entry.get("is_marley")
            if is_marley is not None:
                marley_flags[bool(is_marley)] += 1

            plane = entry.get("plane") or entry.get("plane_id") or entry.get("plane_name")
            if plane is not None:
                plane_counts[str(plane)] += 1

            if len(samples) < sample_count:
                samples.append({**entry, "__source_file": str(npz_path)})

    return ScanResult(
        files_scanned=files_scanned,
        entries_seen=entries_seen,
        interaction_counts=interaction_counts,
        es_flags=es_flags,
        main_track_flags=main_track_flags,
        plane_counts=plane_counts,
        marley_flags=marley_flags,
        samples=samples,
    )


def summarize_counter(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def format_summary(result: ScanResult, as_json: bool) -> str:
    summary = {
        "files_scanned": result.files_scanned,
        "metadata_entries": result.entries_seen,
        "interaction_type_counts": summarize_counter(result.interaction_counts),
        "is_es_counts": summarize_counter(result.es_flags),
        "is_main_track_counts": summarize_counter(result.main_track_flags),
        "is_marley_counts": summarize_counter(result.marley_flags),
        "plane_counts": summarize_counter(result.plane_counts),
        "sample_metadata": result.samples,
    }

    if as_json:
        return json.dumps(summary, indent=2, sort_keys=False)

    lines: List[str] = []
    lines.append(f"Files scanned: {summary['files_scanned']}")
    lines.append(f"Metadata entries: {summary['metadata_entries']}")

    def append_counter(title: str, data: Dict[str, int]) -> None:
        if not data:
            lines.append(f"{title}: (none)")
            return
        counts = ", ".join(f"{key}={value}" for key, value in data.items())
        lines.append(f"{title}: {counts}")

    append_counter("Interaction types", summary["interaction_type_counts"])
    append_counter("is_es", summary["is_es_counts"])
    append_counter("is_main_track", summary["is_main_track_counts"])
    append_counter("is_marley", summary["is_marley_counts"])
    append_counter("Planes", summary["plane_counts"])

    if summary["sample_metadata"]:
        lines.append("\nSample metadata entries:")
        for sample in summary["sample_metadata"]:
            sample_copy = dict(sample)
            source = sample_copy.pop("__source_file", "unknown")
            lines.append(f"- {source}")
            for key, value in sample_copy.items():
                lines.append(f"    {key}: {value}")
            lines.append("")

    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Directory that holds NPZ volume files")
    parser.add_argument(
        "--pattern",
        default="*.npz",
        help="Glob pattern to select files (default: *.npz)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to inspect",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of initial files to skip before processing",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="How many metadata examples to include in the summary",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit summary as JSON for downstream tooling",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages and load failures",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser()

    if not root.exists():
        print(f"Error: {root} does not exist")
        return 1
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        return 1

    result = scan_directory(
        root=root,
        pattern=args.pattern,
        recursive=args.recursive,
        limit=args.limit,
        skip=max(args.skip, 0),
        sample_count=max(args.samples, 0),
        verbose=args.verbose,
    )

    summary_text = format_summary(result, args.json)
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
