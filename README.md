# PW-NER: Persian Wikipedia Named Entity Extraction Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18365950.svg)](https://doi.org/10.5281/zenodo.18365950)

A reproducible pipeline for constructing silver-standard named entity inventories from Persian Wikipedia, designed to support research in Persian natural language processing.

## Overview

PW-NER provides a deterministic, version-controlled extraction pipeline that processes Persian Wikipedia text to generate entity inventories for six entity classes: PER, LOC, ORG, FAC, PRO, and EVENT. The pipeline integrates Hazm normalization and Stanza NER with pinned dependencies to ensure reproducible extraction across environments.

## Features

- Deterministic preprocessing with ZWNJ normalization and compound segmentation
- Pinned dependencies for reproducible execution
- Six entity classes with validated extraction
- Quality control with clean/flagged entity splits
- SHA-256 integrity verification
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

```bash
git clone git@github.com:amirradnia99/persian-ner-pipeline.git
cd persian-ner-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

Place the Farsi Wikipedia snapshot at:

```
data/raw_data.csv
```

Required columns:

- `title`: Wikipedia article title
- `content`: Raw article text
- `link`: Article URL (optional)

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

### Quality Control

```bash
python qc.py \
  --input_dir PW-NER/inventories \
  --output_dir PW-NER/qc \
  --write_raw_copy
```

### Integrity Verification

```bash
python scripts/make_checksums.py \
  --root artifacts \
  --out artifacts/checksums_sha256.txt \
  --verify
```

## Repository Structure

```
persian-ner-pipeline/
├── pipeline.py
├── qc.py
├── requirements.txt
├── CITATION.cff
├── LICENSE
├── README.md
├── artifacts/
├── data/
├── docs/
└── scripts/
    └── make_checksums.py
```

## Dependencies

```
stanza==1.8.2
hazm==0.9.1
beautifulsoup4==4.12.3
pandas==2.2.1
numpy==1.26.4
openpyxl==3.1.2
```

## Citation

```bibtex
@software{radnia2026pwner,
  author = {Radnia, Amir and Keshvari, Saman and Naderi, Hassan},
  title = {PW-NER: Persian Wikipedia Named Entity Extraction Pipeline},
  year = {2026},
  publisher = {Zenodo},
  version = {v1.0.0-silver},
  doi = {10.5281/zenodo.18365950}
}
```

## License

MIT License. Source corpus is external and distributed under its original license.

## Contact

Repository: https://github.com/amirradnia99/persian-ner-pipeline

DOI: https://doi.org/10.5281/zenodo.18365950
