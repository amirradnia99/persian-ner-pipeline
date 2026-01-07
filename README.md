# Persian NER Pipeline (PW-NER)

A **research-grade Persian (Farsi) Named Entity Recognition (NER) pipeline** for large-scale extraction of
**silver-standard entity inventories** from Persian Wikipedia.  
This repository accompanies a survey/resource-style manuscript and provides the **exact code, metadata,
and integrity checks** required for reproducibility.

---

## Overview

This project provides:

- A clean, reproducible **NER pipeline** based on **Hazm normalization + Stanza NER**
- **Silver-standard named entity inventories** extracted from Persian Wikipedia
- Full **data provenance**, **checksum-based integrity verification**, and **citation metadata**
- A repository layout suitable for **Q1 journal artifact evaluation**

The pipeline is designed for **resource construction and analysis**, not supervised model training.

---

## Entity Types

The released inventories include the following entity classes (as produced by Stanza for Persian):

- **PER** – Person  
- **LOC** – Location  
- **ORG** – Organization  
- **FAC** – Facility  
- **PRO** – Product  
- **EVENT** – Event  

Entities are stored as **JSON objects** with text spans and character offsets (relative to normalized text).

---

## Dataset

### Source Corpus

- **Name:** Farsi Wikipedia  
- **Provider:** Amir Pourmand  
- **Platform:** Kaggle  
- **License:** CC0 (Public Domain)  
- **Snapshot date:** 1400/04/25  

The raw corpus is **not redistributed** due to size constraints.

To obtain the dataset:

https://www.kaggle.com/datasets/amirpourmand/fa-wikipedia

Expected input path after download:

```
data/raw_data.csv
```

See `data/README.md` for details.

---

## Preprocessing and NER Pipeline

The released inventories were generated using the following steps:

- HTML tag removal (BeautifulSoup)
- Persian text normalization (Hazm)
- Whitespace normalization
- Named Entity Recognition using **Stanza (fa)**

**Important notes:**

- NER is performed on **normalized natural text**
- No tokenization or stopword removal is applied prior to NER
- Character offsets refer to positions in the **normalized text**, not raw Wikipedia markup
- Inference is executed **CPU-only** for consistency

The full implementation is available in:

```
pipeline.py
```

---

## Artifact Access

The **silver-standard Persian NER inventories** are released via GitHub Releases.

### Latest release (recommended)

https://github.com/amirradnia99/persian-ner-pipeline/releases/tag/v1.0.1-silver

Each release includes:

- Compressed entity inventories (e.g., `PW-NER.rar`)
- `checksums_sha256.txt` for integrity verification
- `manifest.json` describing:
  - Corpus snapshot
  - Pipeline configuration
  - Entity classes and counts

See `artifacts/README.md` for verification instructions.

---

## Reproducibility

This repository provides:

- Exact pipeline implementation used for extraction
- Explicit corpus source and snapshot date
- SHA-256 checksums for released artifacts
- Machine-readable citation metadata (`CITATION.cff`)

The pipeline supports **fault-tolerant execution** with progress tracking for long runs.

---

## Citation

If you use this code or the released entity inventories, please cite the repository.
Citation metadata is provided in:

```
CITATION.cff
```

GitHub automatically exposes citation formats via the **“Cite this repository”** feature.

---

## License

- **Code:** MIT License  
- **Source corpus:** CC0 (via Kaggle)  
- **Derived inventories:** Released as silver-standard research resources

---

## Disclaimer

The released entities are **automatically extracted** and may contain noise or labeling errors.
They are intended for **analysis, benchmarking, and survey research**, not as gold-standard annotations.
