#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline.py — Persian Wikipedia NER Extraction (PW-NER)

CLI-based, reproducible extraction pipeline for generating Persian NER inventories
from a Wikipedia-like CSV using Hazm normalization and Stanza NER.


"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

# NOTE: these heavy deps are imported at module import time intentionally
# so CI/import checks fail fast if the environment is incomplete.
import stanza  # type: ignore

try:
    from hazm import Normalizer  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import Hazm. Install dependencies via requirements.txt."
    ) from e


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOG = logging.getLogger("pwner.pipeline")


# ---------------------------------------------------------------------
# Constants / Contracts
# ---------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
ENTITY_TYPES = ("PER", "LOC", "ORG", "FAC", "PRO", "EVENT")

# Explicit (stable) normalization contract. We filter these keys against the
# hazm.Normalizer signature at runtime to avoid version-mismatch breakage.
DEFAULT_NORMALIZER_CONFIG: Dict[str, Any] = {
    "correct_spacing": True,
    "remove_diacritics": True,
    "remove_specials_chars": False,
    "persian_style": True,
    "persian_numbers": True,
    "remove_extra_spaces": True,
    "unify_chars": True,
    "separate_mi": True,
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Persian NER extraction on Wikipedia content (PW-NER).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # IO
    parser.add_argument("--input", default="data/raw_data.csv", help="Input CSV file")
    parser.add_argument("--output_dir", default="outputs/run_raw", help="Output directory")

    # Column mapping (data contracts vary across dumps)
    parser.add_argument("--title_column", default="title", help="CSV column for article title")
    parser.add_argument("--content_column", default="content", help="CSV column for article content")
    parser.add_argument("--link_column", default="link", help="CSV column for article link (optional)")

    # Processing controls
    parser.add_argument("--chunksize", type=int, default=5000, help="CSV chunksize for streaming reads")
    parser.add_argument("--batch_size", type=int, default=250, help="Mini-batch size inside each chunk")
    parser.add_argument("--log_every", type=int, default=5000, help="Log after this many processed rows")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit (for quick runs)")
    parser.add_argument("--resume", action="store_true", help="Resume from progress.txt if present")

    # Stanza
    parser.add_argument("--stanza_download_dir", default=None, help="Optional Stanza model dir")

    # Which text fields to run NER on
    parser.add_argument(
        "--text_field",
        choices=["title", "content", "both"],
        default="both",
        help="Which text fields to run NER on",
    )

    # Manifest determinism
    parser.add_argument(
        "--no_timestamp",
        action="store_true",
        help="Omit timestamps from run_manifest.json for deterministic manifests",
    )

    # Inventory export
    parser.add_argument(
        "--export_inventories",
        action="store_true",
        help="Export per-class inventories (PER/LOC/ORG/FAC/PRO/EVENT) while running",
    )
    parser.add_argument(
        "--inventories_format",
        choices=["xlsx", "csv"],
        default="xlsx",
        help="Inventory export format",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text to a temp file then os.replace() for atomicity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def read_progress(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return 0


def write_progress(path: Path, value: int) -> None:
    atomic_write_text(path, str(int(value)) + "\n", encoding="utf-8")


def strip_html(text: str) -> str:
    return BeautifulSoup(text or "", "html.parser").get_text()


def build_normalizer() -> Tuple["Normalizer", Dict[str, Any]]:
    """Build hazm.Normalizer with an explicit config, filtered by signature."""
    try:
        import inspect  # local import to keep top-level lightweight
        sig = inspect.signature(Normalizer)
        allowed = set(sig.parameters.keys())
        cfg = {k: v for k, v in DEFAULT_NORMALIZER_CONFIG.items() if k in allowed}
        return Normalizer(**cfg), cfg
    except Exception as e:
        raise RuntimeError("Failed to initialize Hazm Normalizer.") from e


def normalize_fa(text: str, normalizer: "Normalizer") -> str:
    text = normalizer.normalize(text or "")
    return _WS_RE.sub(" ", text).strip()


def ensure_and_standardize_columns(
    df: pd.DataFrame, title_col: str, content_col: str, link_col: str
) -> pd.DataFrame:
    """Rename user-provided columns into the internal contract: title/content/link."""
    missing = []
    for required in (title_col, content_col):
        if required not in df.columns:
            missing.append(required)
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Available: {list(df.columns)[:20]}")

    rename_map = {title_col: "title", content_col: "content"}
    if link_col in df.columns:
        rename_map[link_col] = "link"
    df = df.rename(columns=rename_map)

    if "link" not in df.columns:
        df["link"] = ""

    # Ensure strings (avoid NaN propagation)
    df["title"] = df["title"].astype(str).fillna("")
    df["content"] = df["content"].astype(str).fillna("")
    df["link"] = df["link"].astype(str).fillna("")
    return df


# ---------------------------------------------------------------------
# NER
# ---------------------------------------------------------------------
def build_stanza(download_dir: Optional[str]) -> stanza.Pipeline:
    try:
        return stanza.Pipeline(
            lang="fa",
            processors="tokenize,ner",
            use_gpu=False,
            verbose=False,
            dir=download_dir,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize Stanza pipeline. "
            "If models are missing, run: python -c \"import stanza; stanza.download('fa')\""
        ) from e


def extract_entities(nlp: stanza.Pipeline, text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    doc = nlp(text)
    entities: List[Dict[str, Any]] = []
    for sent in doc.sentences:
        for ent in sent.ents:
            entities.append(
                {
                    "text": ent.text,
                    "type": ent.type,
                    "start_char": int(ent.start_char),
                    "end_char": int(ent.end_char),
                }
            )
    return entities


# ---------------------------------------------------------------------
# Inventory export (streaming-safe)
# ---------------------------------------------------------------------
class InventoryWriter:
    # File basenames must match the released dataset contract
    FILE_BASENAMES = {
        "PER": "pers",
        "LOC": "loc",
        "ORG": "org",
        "FAC": "fac",
        "PRO": "pro",
        "EVENT": "event",
    }

    """
    Stream per-entity-type inventories to disk without keeping them in memory.

    For 'csv': writes <type>.csv with one column "<type>_entities"
    For 'xlsx': writes temp CSVs first, then converts to xlsx at finalize().
    """

    def __init__(self, out_dir: Path, fmt: str) -> None:
        self.out_dir = out_dir
        self.fmt = fmt
        self.tmp_dir = out_dir / "_inventories_tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self._writers: Dict[str, Tuple[Any, csv.writer]] = {}
        self._paths: Dict[str, Path] = {}

        for t in ENTITY_TYPES:
            col = self._col_name(t)
            base = self.FILE_BASENAMES[t]
            path = self.tmp_dir / f"{base}.csv"
            f = open(path, "a", encoding="utf-8", newline="")
            w = csv.writer(f)
            if path.stat().st_size == 0:
                w.writerow([col])
            self._writers[t] = (f, w)
            self._paths[t] = path

    @staticmethod
    def _col_name(t: str) -> str:
        # match your released artifact column naming
        mapping = {
            "PER": "pers_entities",
            "LOC": "loc_entities",
            "ORG": "org_entities",
            "FAC": "fac_entities",
            "PRO": "pro_entities",
            "EVENT": "event_entities",
        }
        return mapping[t]

    def write_from_entities(self, entities: List[Dict[str, Any]]) -> None:
        for ent in entities:
            t = str(ent.get("type", "")).upper()
            if t not in self._writers:
                continue
            text = str(ent.get("text", "")).strip()
            if not text:
                continue
            f, w = self._writers[t]
            w.writerow([text])

    def close(self) -> None:
        for f, _ in self._writers.values():
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass

    def finalize(self) -> None:
        self.close()

        inv_dir = self.out_dir / "inventories"
        inv_dir.mkdir(parents=True, exist_ok=True)

        if self.fmt == "csv":
            # atomic-ish: move files into final dir
            for t, p in self._paths.items():
                base = self.FILE_BASENAMES[t]
                dest = inv_dir / f"{base}.csv"
                os.replace(p, dest)
        else:
            # xlsx: convert each temp CSV to a single-sheet xlsx
            for t, p in self._paths.items():
                df = pd.read_csv(p, encoding="utf-8")
                base = self.FILE_BASENAMES[t]
                dest = inv_dir / f"{base}.xlsx"
                # Write atomically
                fd, tmp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".tmp", dir=str(inv_dir))
                os.close(fd)
                try:
                    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False, sheet_name="Sheet1")
                    os.replace(tmp, dest)
                finally:
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except Exception:
                        pass

        # cleanup tmp dir (best-effort)
        try:
            for t, p in self._paths.items():
                if p.exists():
                    p.unlink()
            self.tmp_dir.rmdir()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Run Manifest
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class RunManifest:
    artifact_name: str
    script: str
    input_path: str
    output_dir: str
    processed_csv: str
    progress_file: str
    text_field: str
    chunksize: int
    batch_size: int
    limit: Optional[int]
    stanza_lang: str
    stanza_processors: str
    stanza_use_gpu: bool
    stanza_download_dir: Optional[str]
    normalizer_config: Dict[str, Any]
    no_timestamp: bool
    generated_at: Optional[str]


def write_manifest(path: Path, manifest: RunManifest) -> None:
    payload = asdict(manifest)
    # Deterministic JSON ordering for stable diffs
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    processed_csv = output_dir / "processed_data.csv"
    processed_tmp = output_dir / "processed_data.csv.tmp"
    progress_path = output_dir / "progress.txt"
    manifest_path = output_dir / "run_manifest.json"

    # Resume offset
    start_row = read_progress(progress_path) if args.resume else 0
    if start_row < 0:
        start_row = 0

    # Build components
    nlp = build_stanza(args.stanza_download_dir)
    normalizer, normalizer_cfg = build_normalizer()

    # Manifest (deterministic option)
    generated_at = None if args.no_timestamp else datetime.utcnow().isoformat(timespec="seconds") + "Z"
    manifest = RunManifest(
        artifact_name="PW-NER",
        script="pipeline.py",
        input_path=str(input_path),
        output_dir=str(output_dir),
        processed_csv=str(processed_csv),
        progress_file=str(progress_path),
        text_field=args.text_field,
        chunksize=int(args.chunksize),
        batch_size=int(args.batch_size),
        limit=args.limit,
        stanza_lang="fa",
        stanza_processors="tokenize,ner",
        stanza_use_gpu=False,
        stanza_download_dir=args.stanza_download_dir,
        normalizer_config=normalizer_cfg,
        no_timestamp=bool(args.no_timestamp),
        generated_at=generated_at,
    )
    write_manifest(manifest_path, manifest)

    # Inventory export
    inv_writer: Optional[InventoryWriter] = None
    if args.export_inventories:
        inv_writer = InventoryWriter(output_dir, fmt=args.inventories_format)

    # Decide output file target
    # - non-resume: write to processed_tmp then atomically replace processed_csv at end
    # - resume: append directly to processed_csv (can't be perfectly atomic without rewriting)
    out_path = processed_csv if args.resume else processed_tmp
    out_mode = "a" if args.resume and processed_csv.exists() else "w"
    write_header = (out_mode == "w")

    processed_rows = 0
    seen_rows = 0  # counts rows read from iterator (including skipped)

    # Streaming CSV reader
    reader = pd.read_csv(
        input_path,
        encoding="utf-8",
        dtype=str,  # stable typing; everything is coerced to string later
        chunksize=args.chunksize,
    )

    with open(out_path, out_mode, encoding="utf-8", newline="") as f_out:
        for chunk in reader:
            # Respect resume offset by discarding rows until start_row reached
            chunk_len = len(chunk)
            if start_row and seen_rows + chunk_len <= start_row:
                seen_rows += chunk_len
                continue

            if start_row and seen_rows < start_row:
                # partial skip within chunk
                cut = start_row - seen_rows
                chunk = chunk.iloc[cut:].copy()
                seen_rows = start_row
            else:
                seen_rows += chunk_len

            # Standardize columns according to mapping
            chunk = ensure_and_standardize_columns(chunk, args.title_column, args.content_column, args.link_column)

            # Apply optional limit
            if args.limit is not None:
                remaining = args.limit - processed_rows
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()

            # Normalize
            chunk["norm_title"] = chunk["title"].map(strip_html).map(lambda t: normalize_fa(t, normalizer))
            chunk["norm_content"] = chunk["content"].map(strip_html).map(lambda t: normalize_fa(t, normalizer))

            # Process in mini-batches (keeps stanza calls bounded)
            for i in range(0, len(chunk), args.batch_size):
                batch = chunk.iloc[i : i + args.batch_size].copy()

                # Run NER
                if args.text_field in ("title", "both"):
                    ents_title = batch["norm_title"].map(lambda t: extract_entities(nlp, t)).tolist()
                    batch["entities_title"] = [json.dumps(e, ensure_ascii=False) for e in ents_title]
                    if inv_writer is not None:
                        for e in ents_title:
                            inv_writer.write_from_entities(e)

                if args.text_field in ("content", "both"):
                    ents_content = batch["norm_content"].map(lambda t: extract_entities(nlp, t)).tolist()
                    batch["entities_content"] = [json.dumps(e, ensure_ascii=False) for e in ents_content]
                    if inv_writer is not None:
                        for e in ents_content:
                            inv_writer.write_from_entities(e)

                cols = ["title", "link"]
                if "entities_title" in batch.columns:
                    cols.append("entities_title")
                if "entities_content" in batch.columns:
                    cols.append("entities_content")

                batch[cols].to_csv(f_out, index=False, header=write_header)
                write_header = False

                processed_rows += len(batch)
                write_progress(progress_path, start_row + processed_rows)

                if processed_rows % args.log_every == 0:
                    total = args.limit if args.limit is not None else None
                    if total is None:
                        LOG.info("Processed %d rows", start_row + processed_rows)
                    else:
                        LOG.info("Processed %d / %d rows", start_row + processed_rows, total)

    # Finalize inventories
    if inv_writer is not None:
        LOG.info("Finalizing inventories (%s)...", args.inventories_format)
        inv_writer.finalize()

    # Atomic finalize for non-resume processed_data.csv
    if not args.resume:
        # Ensure data flushed to disk before swap
        if processed_tmp.exists():
            os.replace(processed_tmp, processed_csv)

    LOG.info("Finished successfully")
    LOG.info("Output directory: %s", output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
