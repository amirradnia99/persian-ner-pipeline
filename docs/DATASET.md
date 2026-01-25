# DATASET: PW-NER (Silver-Standard Persian Named Entity Inventories)

## 1. Summary

**PW-NER** is a silver-standard dataset of Persian (Farsi) named entity inventories automatically extracted from Persian Wikipedia using a reproducible pipeline built on:

- **Hazm** (explicit normalization contract)
- **Stanza NER (fa)** (CPU-only recommended)

This repository does **not** redistribute the raw Persian Wikipedia corpus due to size constraints.
Instead, it releases derived entity inventories with provenance and integrity metadata (manifests + checksums).

---

## 2. What This Dataset Contains

The released dataset consists of **per-class entity inventories** (Excel/CSV) and accompanying metadata.

### Entity classes
The inventories contain the following Stanza entity types:

- **PER** — Person
- **LOC** — Location
- **ORG** — Organization
- **FAC** — Facility
- **PRO** — Product
- **EVENT** — Event

### Typical files (per release asset)
A release asset (e.g., `PW-NER.rar`) is expected to include:

- `pers.xlsx`
- `loc.xlsx`
- `org.xlsx`
- `fac.xlsx`
- `pro.xlsx`
- `event.xlsx`

Additionally, integrity and metadata files are provided via this repository:

- `artifacts/manifest.json`
- `artifacts/checksums_sha256.txt`

---

## 3. Source Corpus (Not Redistributed)

**Corpus:** Farsi Wikipedia (Kaggle)  
**Platform:** Kaggle  
**License:** CC0 (Public Domain)

Expected local path after download:
```
data/raw_data.csv
```

Default required columns:
- `title` — article title
- `content` — article text
- `link` — permanent Wikipedia URL (optional)

Snapshot used for the official release:
- **Kaggle snapshot date:** **1400/04/25**  
See `data/README.md` for details.

---

## 4. Generation Pipeline (End-to-End)

The inventories are produced end-to-end by `pipeline.py` with inventory export enabled:

```bash
python pipeline.py \
  --input data/raw_data.csv \
  --output_dir PW-NER \
  --export_inventories \
  --inventories_format xlsx
```

This produces:
- `PW-NER/processed_data.csv` (intermediate)
- `PW-NER/inventories/*.xlsx` (final dataset tables)

Optional QC (recommended for reuse as “clean” inventories):
```bash
python qc.py --input_dir PW-NER/inventories --output_dir PW-NER/qc
```

---

## 5. Data Schema

PW-NER is inventory-style (silver-standard). Each file contains a single column:

- `pers.xlsx` → `pers_entities`
- `loc.xlsx` → `loc_entities`
- `org.xlsx` → `org_entities`
- `fac.xlsx` → `fac_entities`
- `pro.xlsx` → `pro_entities`
- `event.xlsx` → `event_entities`

Each row is one extracted entity surface form after normalization.

> If you publish QC outputs, `clean/` inventories are recommended for downstream analysis and KG construction.

---

## 6. Limitations

- Automatically extracted (silver-standard)
- Contains noise from Wikipedia templates, lists, and formatting artifacts
- No disambiguation: the same surface form may refer to multiple entities
- Not a gold-standard annotation dataset
