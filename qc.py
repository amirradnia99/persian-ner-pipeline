#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qc.py — Quality Control for PW-NER inventories

CLI-based, deterministic QC over extracted entity inventories stored in .xlsx files.

Typical workflow
1) Run pipeline.py to produce raw inventories (your exporter step)
2) Store per-class inventories as Excel files (e.g., pers.xlsx, org.xlsx, ...)
3) Run this QC to split into:
   - raw/      (normalized copy, optional)
   - clean/    (recommended for reuse)
   - flagged/  (kept for audit)
   - qc_report.json (full audit report)

This QC is rule-based: it flags suspicious entity strings (too long, too many tokens,
digit-only, template phrases, list-like concatenations, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Defaults
# -----------------------------
ARABIC_COMMA = "،"
AND_CONNECTORS = (" و ", " یا ", " & ", " and ", " or ")

DEFAULT_THRESHOLDS = {
    "max_tokens": 7,
    "max_chars": 60,
    "max_digits_ratio": 0.65,
    "max_separators": 3,
    "max_punct_ratio": 0.18,
    "min_arabic_commas_for_list": 2,
    "min_and_connectors_for_list": 3,
    "min_non_digit_chars_when_digits_present": 3,
    "low_diversity_min_len": 14,
    "low_diversity_uniq_ratio": 0.25,
}

DEFAULT_CLASS_FILES = {
    "pers.xlsx": "pers_entities",
    "loc.xlsx": "loc_entities",
    "org.xlsx": "org_entities",
    "fac.xlsx": "fac_entities",
    "pro.xlsx": "pro_entities",
    "event.xlsx": "event_entities",
}


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run deterministic QC on PW-NER Excel inventories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing input .xlsx files. If omitted, --base_dir is used.",
    )
    p.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Legacy: directory containing the .xlsx files directly.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qc_run",
        help="Output directory for QC splits and report.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON file to override thresholds and/or file mapping.",
    )
    p.add_argument(
        "--write_raw_copy",
        action="store_true",
        help="If set, writes a normalized raw copy into output_dir/raw/",
    )
    p.add_argument(
        "--no_timestamp",
        action="store_true",
        help="Omit timestamps from qc_report.json for deterministic outputs.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="How many top clean entities to report per file.",
    )
    return p.parse_args()


# -----------------------------
# Config loading
# -----------------------------
def load_config(path: Optional[Path]) -> Tuple[Dict, Dict]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    mapping = dict(DEFAULT_CLASS_FILES)

    if path is None:
        return thresholds, mapping

    cfg = json.loads(path.read_text(encoding="utf-8"))
    if "thresholds" in cfg and isinstance(cfg["thresholds"], dict):
        thresholds.update(cfg["thresholds"])
    if "files" in cfg and isinstance(cfg["files"], dict):
        mapping.update(cfg["files"])
    return thresholds, mapping


# -----------------------------
# Rule helpers
# -----------------------------
# NOTE: original file had an invalid Python string here; fixed using a triple-quoted raw string.
_PUNCT_RE = re.compile(r"""[\.,;:!\?\-\(\)\[\]\{\}«»"'…ـ،]""")
_DIGIT_RE = re.compile(r"[0-9۰-۹]")
_WS_RE = re.compile(r"\s+")


def normalize_entity(s: str) -> str:
    s = (s or "").strip()
    s = _WS_RE.sub(" ", s)
    return s


def token_count(s: str) -> int:
    if not s:
        return 0
    return len(s.split())


def punct_ratio(s: str) -> float:
    if not s:
        return 0.0
    punct = len(_PUNCT_RE.findall(s))
    return punct / max(len(s), 1)


def separator_count(s: str) -> int:
    return s.count(ARABIC_COMMA) + s.count(",") + s.count("/") + s.count("|")


def digit_stats(s: str) -> Tuple[int, int]:
    digits = len(_DIGIT_RE.findall(s))
    non_digits = max(len(s) - digits, 0)
    return digits, non_digits


_YEAR_ONLY_RE = re.compile(r"^(?:[12][0-9]{3}|[۱۲][۰-۹]{3})$")


def year_only(s: str) -> bool:
    return bool(_YEAR_ONLY_RE.match(s))


def has_list_pattern(s: str, thresholds: Dict) -> bool:
    if separator_count(s) >= int(thresholds["min_arabic_commas_for_list"]):
        return True
    and_hits = sum(1 for c in AND_CONNECTORS if c in f" {s} ")
    return and_hits >= int(thresholds["min_and_connectors_for_list"])


_GENERIC_PREFIXES = (
    "شهر ",
    "استان ",
    "کشور ",
    "منطقه ",
    "روستا ",
    "بندر ",
    "فرودگاه ",
    "دانشگاه ",
    "سازمان ",
    "شرکت ",
)


def is_generic_template_phrase(s: str) -> bool:
    return any(s.startswith(pref) for pref in _GENERIC_PREFIXES)


_DIR_WORDS = ("شمال", "جنوب", "شرق", "غرب")


def is_direction_duplication(s: str) -> bool:
    parts = s.split()
    if len(parts) == 2 and parts[0] in _DIR_WORDS and parts[1] in _DIR_WORDS:
        return parts[0] == parts[1]
    return False


def low_char_diversity(s: str, thresholds: Dict) -> bool:
    s2 = s.replace(" ", "")
    if len(s2) < int(thresholds["low_diversity_min_len"]):
        return False
    uniq = len(set(s2))  # set used only for counting; no ordering leak
    ratio = uniq / max(len(s2), 1)
    return ratio < float(thresholds["low_diversity_uniq_ratio"])


def loc_city_placeholder(s: str) -> bool:
    return s.startswith("شهر ") and len(s.split()) == 2


def loc_direction_plus_country(s: str) -> bool:
    parts = s.split()
    return len(parts) == 2 and parts[0] in _DIR_WORDS


def loc_city_plus_country(s: str) -> bool:
    parts = s.split()
    return len(parts) == 2 and parts[0] in ("شهر", "استان")


def loc_bare_direction(s: str) -> bool:
    return s in _DIR_WORDS


def pro_template_pronoun(s: str) -> bool:
    return s.startswith("او ") or s.startswith("این ")


# -----------------------------
# QC Core
# -----------------------------
RULES_ORDER = [
    "too_many_tokens",
    "too_long",
    "digits_only_or_near",
    "year_only",
    "list_and_connectors",
    "generic_template_phrase",
    "low_char_diversity",
    "direction_duplication",
    "loc_direction_plus_country",
    "loc_city_plus_country",
    "loc_city_placeholder",
    "loc_bare_direction",
    "pro_template_pronoun",
]


def ordered_rule_counts(counter: Counter) -> Dict[str, int]:
    """Return rule counts as an ordered dict for deterministic JSON output."""
    ordered: Dict[str, int] = {}
    for r in RULES_ORDER:
        if r in counter:
            ordered[r] = int(counter[r])
    for r in sorted(k for k in counter.keys() if k not in ordered):
        ordered[r] = int(counter[r])
    return ordered


def ordered_examples(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Order example lists by RULES_ORDER for deterministic JSON output."""
    ordered: Dict[str, List[str]] = {}
    for r in RULES_ORDER:
        if r in examples:
            ordered[r] = list(examples[r])
    for r in sorted(k for k in examples.keys() if k not in ordered):
        ordered[r] = list(examples[r])
    return ordered


def apply_rules(s: str, thresholds: Dict, context: str) -> List[str]:
    rules: List[str] = []

    if token_count(s) > int(thresholds["max_tokens"]):
        rules.append("too_many_tokens")
    if len(s) > int(thresholds["max_chars"]):
        rules.append("too_long")

    digits, non_digits = digit_stats(s)
    if digits > 0:
        if non_digits < int(thresholds["min_non_digit_chars_when_digits_present"]):
            rules.append("digits_only_or_near")
        else:
            ratio = digits / max(len(s), 1)
            if ratio >= float(thresholds["max_digits_ratio"]):
                rules.append("digits_only_or_near")

    if year_only(s):
        rules.append("year_only")

    if has_list_pattern(s, thresholds):
        rules.append("list_and_connectors")

    if is_generic_template_phrase(s):
        rules.append("generic_template_phrase")

    if low_char_diversity(s, thresholds):
        rules.append("low_char_diversity")

    if is_direction_duplication(s):
        rules.append("direction_duplication")

    if context == "loc":
        if loc_direction_plus_country(s):
            rules.append("loc_direction_plus_country")
        if loc_city_placeholder(s):
            rules.append("loc_city_placeholder")
        if loc_city_plus_country(s):
            rules.append("loc_city_plus_country")
        if loc_bare_direction(s):
            rules.append("loc_bare_direction")

    if context == "pro":
        if pro_template_pronoun(s):
            rules.append("pro_template_pronoun")

    return rules


# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class FileReport:
    file: str
    rows_total: int
    rows_flagged: int
    rows_clean: int
    entity_column: str
    class_column: Optional[str]
    provenance_doc_col: Optional[str]
    provenance_list_col: Optional[str]
    rule_counts: Dict[str, int]
    examples_by_rule: Dict[str, List[str]]
    top_entities_clean: List[Dict[str, int]]


# -----------------------------
# IO + Processing
# -----------------------------
def infer_context_from_filename(fn: str) -> str:
    return Path(fn).stem.lower()


def safe_read_excel(path: Path) -> pd.DataFrame:
    """Read Excel with stable typing and without pandas NA coercion."""
    return pd.read_excel(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="openpyxl",
    )


def write_excel(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text atomically (temp file then replace)."""
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


def main() -> None:
    args = parse_args()

    thresholds, mapping = load_config(Path(args.config)) if args.config else (dict(DEFAULT_THRESHOLDS), dict(DEFAULT_CLASS_FILES))

    # Determine input directory
    if args.input_dir:
        in_dir = Path(args.input_dir)
    elif args.base_dir:
        in_dir = Path(args.base_dir)
    else:
        raise SystemExit("ERROR: Provide --input_dir (recommended) or --base_dir (legacy).")

    if not in_dir.exists():
        raise SystemExit(f"ERROR: input directory not found: {in_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = out_dir / "raw"
    clean_dir = out_dir / "clean"
    flagged_dir = out_dir / "flagged"

    report = {
        "generated_at": None if args.no_timestamp else datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "base_dir": str(out_dir),
        "thresholds": thresholds,
        "files": {},
        "overall": {},
    }

    overall_total = 0
    overall_flagged = 0
    overall_clean = 0
    overall_rule_counts = Counter()

    # Process each expected file if present (sorted for determinism)
    for filename in sorted(mapping.keys()):
        entity_col = mapping[filename]
        xlsx_path = in_dir / filename
        if not xlsx_path.exists():
            continue

        ctx = infer_context_from_filename(filename)
        df = safe_read_excel(xlsx_path)

        if entity_col not in df.columns:
            raise SystemExit(
                f"ERROR: {filename} missing expected entity column '{entity_col}'. Columns: {list(df.columns)}"
            )

        # Normalize entity strings (stable conversion + strip + whitespace normalize)
        df[entity_col] = df[entity_col].astype(str).map(normalize_entity)

        if args.write_raw_copy:
            write_excel(df[[entity_col]].copy(), raw_dir / filename)

        # Apply rules
        rule_hits: List[List[str]] = []
        first_rule: List[str] = []
        for s in df[entity_col].tolist():
            s2 = normalize_entity(s)
            rules = apply_rules(s2, thresholds, ctx)
            rule_hits.append(rules)
            first_rule.append(rules[0] if rules else "")

        df["_qc_rules"] = rule_hits
        df["_qc_flag"] = df["_qc_rules"].map(lambda r: len(r) > 0)
        df["_qc_first_rule"] = first_rule

        total = int(len(df))
        flagged = int(df["_qc_flag"].sum())
        clean = total - flagged

        # Counts + examples
        rule_counts = Counter()
        examples = defaultdict(list)

        for s, rules in zip(df[entity_col].tolist(), rule_hits):
            for r in rules:
                rule_counts[r] += 1
                if len(examples[r]) < 8 and s not in examples[r]:
                    examples[r].append(s)

        overall_rule_counts.update(rule_counts)

        # Split + write
        df_clean = df[df["_qc_flag"] == False][[entity_col]].copy()
        df_flagged = df[df["_qc_flag"] == True][[entity_col, "_qc_first_rule", "_qc_rules"]].copy()

        write_excel(df_clean, clean_dir / filename)
        write_excel(df_flagged, flagged_dir / filename)

        overall_total += total
        overall_flagged += flagged
        overall_clean += clean

        # Top entities (clean)
        top = []
        if len(df_clean) > 0:
            vc = df_clean[entity_col].value_counts().head(int(args.top_k))
            for ent, cnt in vc.items():
                top.append({"entity": str(ent), "count": int(cnt)})

        fr = FileReport(
            file=filename,
            rows_total=total,
            rows_flagged=flagged,
            rows_clean=clean,
            entity_column=entity_col,
            class_column=None,
            provenance_doc_col=None,
            provenance_list_col=None,
            rule_counts=ordered_rule_counts(rule_counts),
            examples_by_rule=ordered_examples(dict(examples)),
            top_entities_clean=top,
        )

        report["files"][filename] = asdict(fr)

    report["overall"] = {
        "total_rows": int(overall_total),
        "flagged_rows": int(overall_flagged),
        "clean_rows": int(overall_clean),
        "rules": ordered_rule_counts(overall_rule_counts),
    }

    atomic_write_text(
        out_dir / "qc_report.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("QC completed.")
    print(f"Output directory: {out_dir}")
    print(f"- clean/:   {clean_dir}")
    print(f"- flagged/: {flagged_dir}")
    if args.write_raw_copy:
        print(f"- raw/:     {raw_dir}")
    print(f"- qc_report.json: {out_dir / 'qc_report.json'}")


if __name__ == "__main__":
    main()
