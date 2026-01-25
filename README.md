
# Persian NER Pipeline (PW-NER)

A research-grade Persian (Farsi) Named Entity Recognition (NER) pipeline and dataset for large-scale extraction of **silver-standard named entity inventories** from Persian Wikipedia.

This repository is designed as a **Q1 journal-facing research artifact**: deterministic CLI tools, pinned dependencies, integrity metadata (manifests + SHA-256), and citation support.

---

## What this repo is (and isn’t)

### It **is**
- A reproducible extraction pipeline (`pipeline.py`)
- A deterministic QC stage (`qc.py`)
- Release metadata & integrity artifacts (`artifacts/`)
- Documentation for dataset, QC, and full reproduction (`docs/`)
- A citable research artifact (`CITATION.cff`)

### It **is not**
- A hosted copy of the raw Persian Wikipedia corpus (too large; obtained externally)
- A gold-standard labeled NER dataset (outputs are **silver-standard**)

---

## Quickstart

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Provide the corpus
Download the referenced Persian Wikipedia snapshot and place it as:
```
data/raw_data.csv
```
See `data/README.md` for the exact input contract (columns, encoding, snapshot date).

### 3) Run extraction + export inventories (end-to-end)
```bash
python pipeline.py   --input data/raw_data.csv   --output_dir PW-NER   --chunksize 5000   --batch_size 250   --text_field both   --export_inventories   --inventories_format xlsx
```

For deterministic manifests (no timestamps):
```bash
python pipeline.py ... --no_timestamp
```

Outputs are written under `PW-NER/`. Directory contract: `outputs/README.md`.

### 4) Run QC (optional, recommended)
```bash
python qc.py   --input_dir PW-NER/inventories   --output_dir PW-NER/qc   --write_raw_copy
```

For deterministic QC reports:
```bash
python qc.py ... --no_timestamp
```

---

## Entity types

Entities are produced using **Stanza (fa)** and grouped into six classes:

- **PER** — Person
- **LOC** — Location
- **ORG** — Organization
- **FAC** — Facility
- **PRO** — Product
- **EVENT** — Event

All released entities are automatically extracted (silver-standard).

---

## Repository structure

```
persian-ner-pipeline/
├── pipeline.py
├── qc.py
├── requirements.txt
├── requirements-dev.txt            # optional
├── scripts/
│   └── make_checksums.py
├── artifacts/
│   ├── manifest.json
│   ├── checksums_sha256.txt
│   └── README.md
├── data/
│   └── README.md                   # input corpus contract
├── docs/
│   ├── REPRODUCIBILITY.md
│   ├── DATASET.md
│   ├── QC.md
│   └── CHANGELOG.md
└── outputs/
    └── README.md                   # runtime output layout (not tracked)
```

---

## Dataset access (releases + Zenodo)

The PW-NER inventories are distributed via:
- GitHub Releases
- Zenodo (DOI-backed archival)

Integrity verification:
- `artifacts/checksums_sha256.txt`
- `scripts/make_checksums.py`

---

## Citation

Use GitHub’s “Cite this repository” button or the metadata in:
- `CITATION.cff`

### **DOI:**

The dataset and associated software artifacts are archived on Zenodo:

- **Concept DOI**: [https://doi.org/10.5281/zenodo.18365950](https://doi.org/10.5281/zenodo.18365950)  
  *This DOI represents the entire dataset collection and will always resolve to the latest version.*

- **Version DOI**: [https://doi.org/10.5281/zenodo.18365951](https://doi.org/10.5281/zenodo.18365951)  
  *This DOI corresponds to the specific release of this version (v1.0.0).*

---

## License

- **Code**: MIT (see `LICENSE`)
- **Source corpus**: CC0 (external, via Kaggle)
- **Derived inventories**: silver-standard research artifact (see Zenodo record for the release)

---

## Disclaimer

Automatically generated outputs may contain noise and boundary errors. This is intended for large-scale analysis, benchmarking, KG construction, and resource/survey research — not as a gold-standard annotation dataset.
