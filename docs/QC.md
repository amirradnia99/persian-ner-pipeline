# Quality Control (QC) Protocol — PW-NER

This document defines the deterministic Quality Control (QC) stage applied to PW-NER inventories and the meaning of the resulting `clean/` and `flagged/` splits.

QC exists because the inventories are **silver-standard** outputs from automatic NER over Wikipedia text, which contains templates, lists, and formatting artifacts that can produce noisy entity strings.

---

## 1. Inputs and Outputs

### Input
QC is applied to the **per-class inventory files** produced by `pipeline.py` when run with `--export_inventories`.

Expected input directory example:
```
PW-NER/inventories/
  pers.xlsx
  loc.xlsx
  org.xlsx
  fac.xlsx
  pro.xlsx
  event.xlsx
```

### Output
QC produces:

```
PW-NER/qc/
  clean/
    pers.xlsx
    loc.xlsx
    org.xlsx
    fac.xlsx
    pro.xlsx
    event.xlsx

  flagged/
    pers.xlsx
    loc.xlsx
    org.xlsx
    fac.xlsx
    pro.xlsx
    event.xlsx

  qc_report.json
```

- `clean/` contains entities that pass all QC rules.
- `flagged/` contains entities rejected by one or more rules.
- `qc_report.json` captures thresholds, per-class statistics, rule counts, and examples.

---

## 2. How to Run (Deterministic)

```bash
python qc.py \
  --input_dir PW-NER/inventories \
  --output_dir PW-NER/qc \
  --write_raw_copy
```

For a deterministic report (no timestamps):
```bash
python qc.py ... --no_timestamp
```

---

## 3. Philosophy (What QC Is / Is Not)

QC is **not** gold-standard annotation.
QC is a deterministic, heuristic filtering stage that aims to remove *obviously malformed* strings that are common in Wikipedia-derived extraction.

Design goals:
- deterministic (same inputs → same split)
- auditable (rule counts + examples)
- conservative (keep most plausible entities)

---

## 4. Rule Families (High-Level)

QC flags entities that match any of these families (see `qc.py` for exact definitions):

A) Length and token constraints (overlong / too many tokens)  
B) Numeric noise (year-only, near-digit entities)  
C) List/enumeration artifacts (“X و Y و Z …”, comma-chains)  
D) Template / placeholder phrases (“شهر …”, “استان …”, etc.)  
E) Low character diversity / garbage strings  
F) Directional artifacts (mostly LOC/FAC)  
G) Product/media pronoun templates (PRO)

---

## 5. Reporting and Auditability

`qc_report.json` includes:
- generation metadata (optional timestamp)
- thresholds used
- per-class stats (total / clean / flagged)
- rule-level violation counts (stable ordering)
- small examples per rule

If you publish QC outputs, publish:
- `qc_report.json`
- checksums for `clean/` and `flagged/`
- a QC manifest describing the QC stage (optional but recommended)

---

## 6. Recommended Dataset for Reuse

For downstream reuse, prefer:
- **`clean/` inventories** for analysis and KG construction
- **raw inventories** if maximizing recall matters more than precision
- **flagged** only for diagnostics and rule refinement
