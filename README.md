# PW-NER: Persian Wikipedia Named Entity Extraction Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18365950.svg)](https://doi.org/10.5281/zenodo.18365950)

A reproducible pipeline for constructing silver-standard named entity inventories from Persian Wikipedia, designed to support research in Persian natural language processing.

## Overview

PW-NER is a deterministic, version-controlled Persian Named Entity Recognition extraction pipeline. It processes Persian Wikipedia text and produces entity inventories for six entity classes:

- PER: Person
- LOC: Location
- ORG: Organization
- FAC: Facility
- PRO: Product
- EVENT: Event

The pipeline integrates Hazm normalization and Stanza NER with pinned dependencies for reproducible extraction.

## Features

- Deterministic preprocessing with ZWNJ normalization
- Compound segmentation support
- Pinned dependencies
- Six entity classes
- Quality control with clean/flagged splits
- SHA-256 artifact verification
- Versioned DOI release

## Entity Statistics

| Class | Description | Count |
|---|---|---:|
| PER | Person | 765,974 |
| LOC | Location | 294,747 |
| ORG | Organization | 211,018 |
| FAC | Facility | 97,162 |
| PRO | Product | 41,088 |
| EVENT | Event | 39,595 |

## Installation

Requirements:
- Python 3.10+
- pip

```bash
git clone git@github.com:amirradnia99/persian-ner-pipeline.git
cd persian-ner-pipeline
python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preparation

Place the Farsi Wikipedia snapshot at:

```text
data/raw_data.csv
```

Required columns:

- `title`: Wikipedia article title
- `content`: Raw article text
- `link`: Article URL (optional)

Source:
https://www.kaggle.com/datasets/amirpourmand/fa-wikipedia

## Usage

### Extraction

```bash
python pipeline.py \
  --input data/raw_data.csv \
  --output_dir PW-NER \
  --chunksize 5000 \
  --batch_size 250 \
  --text_field both \
  --export_inventories \
  --inventories_format xlsx
```

Deterministic execution:

```bash
python pipeline.py --input data/raw_data.csv --output_dir PW-NER --no_timestamp
```

### Quality Control

```bash
python qc.py --input_dir PW-NER/inventories --output_dir PW-NER/qc --write_raw_copy
```

### Integrity Verification

```bash
python scripts/make_checksums.py --root artifacts --out artifacts/checksums_sha256.txt --verify
```

## Output Structure

```text
PW-NER/
├── inventories/
│   ├── pers.xlsx
│   ├── loc.xlsx
│   ├── org.xlsx
│   ├── fac.xlsx
│   ├── pro.xlsx
│   └── event.xlsx
├── qc/
│   ├── clean/
│   ├── flagged/
│   └── qc_report.json
├── processed_data.csv
├── run_manifest.json
└── progress.txt
```

## Repository Structure

```text
persian-ner-pipeline/
├── pipeline.py
├── qc.py
├── requirements.txt
├── requirements-dev.txt
├── CITATION.cff
├── LICENSE
├── README.md
├── artifacts/
├── data/
├── docs/
└── scripts/
    └── make_checksums.py
```

## Reproducibility

- Deterministic execution using `--no_timestamp`
- Exact dependency versions in `requirements.txt`
- SHA-256 checksum verification
- Versioned Zenodo DOI release

## Dependencies

```text
stanza==1.8.2
hazm==0.9.1
beautifulsoup4==4.12.3
pandas==2.2.1
numpy==1.26.4
openpyxl==3.1.2
```

## Quality Control Rules

QC removes noisy entities using deterministic rules:

- Length and token constraints
- Numeric-only entities
- Enumeration artifacts
- Template and placeholder phrases
- Low character diversity
- Directional LOC/FAC artifacts
- Product/media pronoun templates

Outputs:

- `clean/`: validated entities
- `flagged/`: entities requiring inspection
- `qc_report.json`: audit statistics

## Citation

```bibtex
@software{radnia2026pwner,
  author = {Radnia, Amir and Keshvari, Saman and Naderi, Hassan},
  title = {PW-NER: Persian Wikipedia Named Entity Extraction Pipeline},
  year = {2026},
  publisher = {Zenodo},
  version = {v1.0.0-silver},
  doi = {10.5281/zenodo.18365950},
  url = {https://github.com/amirradnia99/persian-ner-pipeline}
}
```

## License

- Code: MIT License
- Source corpus: external dataset under original license
- Derived inventories: Silver-standard research artifact

## Contact

Repository:
https://github.com/amirradnia99/persian-ner-pipeline

DOI:
https://doi.org/10.5281/zenodo.18365950
