# PW-NER: Persian Wikipedia Named Entity Extraction Pipeline


[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.18365950-blue)](https://doi.org/10.5281/zenodo.18365950)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
A reproducible pipeline for constructing silver-standard named entity inventories from Persian Wikipedia, designed to support research in Persian natural language processing.

## Overview

PW-NER provides a deterministic, version-controlled extraction pipeline that processes Persian Wikipedia text to generate entity inventories for six entity classes: PER (Person), LOC (Location), ORG (Organization), FAC (Facility), PRO (Product), and EVENT. The pipeline integrates Hazm normalization and Stanza NER with pinned dependencies to ensure reproducible extraction across environments.

## Repository Purpose

This repository serves as the companion implementation for research on Persian Named Entity Recognition. It provides:

- A reproducible extraction pipeline with deterministic preprocessing
- Silver-standard entity inventories for Persian NLP research
- Quality control filtering for entity validation
- Integrity metadata and checksums for verification
- Citable, versioned artifacts with persistent DOIs

## Features

- Deterministic preprocessing with ZWNJ normalization and compound segmentation
- Pinned dependencies for environment reproducibility
- Six entity classes with validated extraction
- Quality control with clean/flagged entity splits
- SHA-256 integrity verification for artifacts
- Versioned DOI for citability (10.5281/zenodo.18365950)

## Entity Statistics

| Class | Description | Count |
|-------|-------------|-------|
| PER | Person | 765,974 |
| LOC | Location | 294,747 |
| ORG | Organization | 211,018 |
| FAC | Facility | 97,162 |
| PRO | Product | 41,088 |
| EVENT | Event | 39,595 |

## Installation

### Requirements

- Python 3.10 or higher
- pip package manager

### Setup

Clone the repository and install dependencies:
```bash
git clone git@github.com:amirradnia99/persian-ner-pipeline.git
cd persian-ner-pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preparation

Download the Farsi Wikipedia snapshot from Kaggle and place it as:

```
data/raw_data.csv
```

**Required columns:**
- `title`: Wikipedia article title
- `content`: Raw article text (may include HTML)
- `link`: Article URL (optional)

*Source: [https://www.kaggle.com/datasets/amirpourmand/fa-wikipedia](https://www.kaggle.com/datasets/amirpourmand/fa-wikipedia)*

## Usage

### End-to-End Extraction

Run the complete extraction pipeline:

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

For deterministic manifests (no timestamps):

```bash
python pipeline.py --input data/raw_data.csv --output_dir PW-NER --no_timestamp
```

### Quality Control

Run QC to filter and validate entities:

```bash
python qc.py \
  --input_dir PW-NER/inventories \
  --output_dir PW-NER/qc \
  --write_raw_copy
```

For deterministic QC reports:

```bash
python qc.py --input_dir PW-NER/inventories --output_dir PW-NER/qc --no_timestamp
```

### Integrity Verification

Verify artifact integrity:

```bash
python scripts/make_checksums.py \
  --root artifacts \
  --out artifacts/checksums_sha256.txt \
  --verify
```

## Output Structure

```
PW-NER/
â”œâ”€â”€ inventories/
â”‚   â”œâ”€â”€ pers.xlsx        # Person entities
â”‚   â”œâ”€â”€ loc.xlsx         # Location entities
â”‚   â”œâ”€â”€ org.xlsx         # Organization entities
â”‚   â”œâ”€â”€ fac.xlsx         # Facility entities
â”‚   â”œâ”€â”€ pro.xlsx         # Product entities
â”‚   â””â”€â”€ event.xlsx       # Event entities
â”œâ”€â”€ qc/
â”‚   â”œâ”€â”€ clean/           # Filtered entities (recommended for reuse)
â”‚   â”œâ”€â”€ flagged/         # Flagged entities (for audit)
â”‚   â””â”€â”€ qc_report.json   # QC statistics and rule counts
â”œâ”€â”€ processed_data.csv   # Intermediate NER annotations
â”œâ”€â”€ run_manifest.json    # Pipeline configuration
â””â”€â”€ progress.txt         # Processing progress
```

## Repository Structure

```
persian-ner-pipeline/
â”œâ”€â”€ pipeline.py              # Main extraction pipeline
â”œâ”€â”€ qc.py                    # Quality Control module
â”œâ”€â”€ requirements.txt         # Production dependencies (pinned)
â”œâ”€â”€ requirements-dev.txt     # Development dependencies
â”œâ”€â”€ CITATION.cff             # Citation metadata
â”œâ”€â”€ LICENSE                  # MIT License
â”œâ”€â”€ README.md                # Documentation
â”œâ”€â”€ artifacts/
â”‚   â”œâ”€â”€ manifest.json        # Release metadata
â”‚   â”œâ”€â”€ checksums_sha256.txt # Integrity verification
â”‚   â””â”€â”€ README.md            # Artifacts documentation
â”œâ”€â”€ data/
â”‚   â””â”€â”€ README.md            # Input corpus contract
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ REPRODUCIBILITY.md   # Reproduction protocol
â”‚   â”œâ”€â”€ DATASET.md           # Dataset documentation
â”‚   â”œâ”€â”€ QC.md                # QC protocol
â”‚   â””â”€â”€ CHANGELOG.md         # Version history
â””â”€â”€ scripts/
    â””â”€â”€ make_checksums.py    # Checksum generation
```

## Reproducibility

This repository is designed as a reproducible research artifact with:

- **Deterministic execution:** `--no_timestamp` flag for reproducible manifests
- **Pinned dependencies:** Exact package versions in `requirements.txt`
- **Integrity verification:** SHA-256 checksums for release artifacts
- **Versioned DOI:** Permanent, citable archive on Zenodo

## Key Dependencies

```
stanza==1.8.2
hazm==0.9.1
beautifulsoup4==4.12.3
pandas==2.2.1
numpy==1.26.4
openpyxl==3.1.2
```

## Quality Control Rules

QC applies deterministic filtering to remove noisy entities based on:

- Length and token constraints
- Numeric noise (year-only, digit-heavy entities)
- List/enumeration artifacts
- Template/placeholder phrases
- Low character diversity
- Directional artifacts (LOC/FAC)
- Product/media pronoun templates (PRO)

The QC process produces:
- `clean/`: Entities passing all QC rules (recommended for reuse)
- `flagged/`: Entities rejected by one or more rules
- `qc_report.json`: Audit report with rule counts and examples

## Citation

If you use this software or the PW-NER dataset, please cite:

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

- **Code:** MIT License (see `LICENSE`)
- **Source corpus:** CC0 (external, via Kaggle)
- **Derived inventories:** Silver-standard research artifact (see Zenodo record)

## Contact

- **Issues:** https://github.com/amirradnia99/persian-ner-pipeline/issues
- **Email:** amir_radnia@cmps2.iust.ac.ir

## Acknowledgments

This work was supported by the Iran University of Science and Technology (IUST), Tehran, Iran.

**Repository:** https://github.com/amirradnia99/persian-ner-pipeline  
**DOI:** https://doi.org/10.5281/zenodo.18365950
