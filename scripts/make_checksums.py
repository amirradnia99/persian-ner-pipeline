#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/make_checksums.py

Deterministically generate and/or verify SHA-256 checksums for release artifacts.

Why this exists:
- In Q1-facing research artifacts, checksums must be reproducible and not "hand-made".
- This script writes checksums in a stable order and stable format.

Usage
-----
# Generate checksums for all files in artifacts/ (recommended)
python scripts/make_checksums.py --root artifacts --out artifacts/checksums_sha256.txt

# Verify against an existing checksums file
python scripts/make_checksums.py --root artifacts --out artifacts/checksums_sha256.txt --verify

# Only hash specific files
python scripts/make_checksums.py --files artifacts/PW-NER.rar artifacts/manifest.json --out artifacts/checksums_sha256.txt

Notes
-----
- Output is sorted by *relative path* for deterministic diffs.
- The output file itself is excluded from hashing to avoid self-reference.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate/verify deterministic SHA-256 checksums for artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", type=str, default=None, help="Root directory to hash recursively.")
    p.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit list of files to hash (overrides --root).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="checksums_sha256.txt",
        help="Output checksum file path.",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Verify current files against the checksums in --out (no rewrite if OK).",
    )
    p.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Optional globs (relative to root) to exclude when using --root.",
    )
    return p.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def collect_files(root: Path, out_path: Path, excludes: List[str] | None) -> List[Path]:
    # Deterministic collection: sorted by relative path.
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        # Exclude the output file itself (avoid self-reference)
        if p.resolve() == out_path.resolve():
            continue
        files.append(p)

    if excludes:
        filtered = []
        for p in files:
            rel = p.relative_to(root).as_posix()
            # Simple glob-style excludes
            if any(Path(rel).match(g) for g in excludes):
                continue
            filtered.append(p)
        files = filtered

    files.sort(key=lambda x: x.relative_to(root).as_posix())
    return files


def format_lines(pairs: Iterable[Tuple[str, str]]) -> str:
    # Keep a stable, grep-friendly format:
    # <relative_path><two_spaces><SHA256_UPPER>
    return "\n".join([f"{name}  {digest}" for name, digest in pairs]) + "\n"


def parse_existing(out_path: Path) -> List[Tuple[str, str]]:
    if not out_path.exists():
        raise FileNotFoundError(f"Checksum file not found: {out_path}")
    pairs: List[Tuple[str, str]] = []
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Expected: "<name>  <hash>"
        if "  " not in line:
            raise ValueError(f"Invalid checksum line (missing double-space separator): {line!r}")
        name, digest = line.split("  ", 1)
        pairs.append((name.strip(), digest.strip().upper()))
    return pairs


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)

    if args.files:
        files = [Path(f) for f in args.files]
        for f in files:
            if not f.exists() or not f.is_file():
                print(f"ERROR: file not found: {f}", file=sys.stderr)
                return 2
        # Determine a root for relative paths (common parent) for stable names
        root = Path.cwd()
        try:
            root = Path(Path(*files[0].parts[: len(files[0].parts) - 1]))  # parent
        except Exception:
            root = Path.cwd()

        # Prefer --root if provided
        if args.root:
            root = Path(args.root)
    else:
        if not args.root:
            print("ERROR: provide either --root or --files", file=sys.stderr)
            return 2
        root = Path(args.root)
        if not root.exists() or not root.is_dir():
            print(f"ERROR: root directory not found: {root}", file=sys.stderr)
            return 2
        files = collect_files(root, out_path=out_path, excludes=args.exclude)

    # Compute current checksums
    pairs: List[Tuple[str, str]] = []
    for p in files:
        try:
            rel = p.relative_to(root).as_posix()
        except Exception:
            rel = p.name
        pairs.append((rel, sha256_file(p)))

    # Deterministic ordering by name
    pairs.sort(key=lambda x: x[0])

    if args.verify:
        existing = parse_existing(out_path)
        existing_map = {n: h for n, h in existing}
        current_map = {n: h for n, h in pairs}

        missing = sorted(set(existing_map) - set(current_map))
        extra = sorted(set(current_map) - set(existing_map))
        mismatched = sorted(n for n in existing_map.keys() & current_map.keys() if existing_map[n] != current_map[n])

        if missing or extra or mismatched:
            if missing:
                print("MISSING files (present in checksum file but not found now):", file=sys.stderr)
                for n in missing:
                    print(f"  - {n}", file=sys.stderr)
            if extra:
                print("EXTRA files (present now but not in checksum file):", file=sys.stderr)
                for n in extra:
                    print(f"  - {n}", file=sys.stderr)
            if mismatched:
                print("MISMATCHED hashes:", file=sys.stderr)
                for n in mismatched:
                    print(f"  - {n}: expected {existing_map[n]}, got {current_map[n]}", file=sys.stderr)
            return 1

        print("OK: all checksums match.")
        return 0

    # Write output deterministically
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(format_lines(pairs), encoding="utf-8")
    print(f"Wrote checksums: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
