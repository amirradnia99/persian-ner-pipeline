# Persian NER Pipeline (Wikipedia-based)

## Overview
This repository provides a **reproducible research pipeline** for large-scale **Named Entity Recognition (NER)** on Persian (Farsi) text, with a specific focus on **Wikipedia-based corpora**.

The pipeline was developed to support **survey and resource-oriented research** by producing **silver-standard entity inventories** that can be used for lexicon construction, weak supervision, coverage analysis, and downstream NLP research.

This repository accompanies a **Q1 journal survey paper** and follows **research software and reproducibility standards** expected by top-tier venues.

---

## Data Source

### Primary Corpus
- **Dataset**: *Farsi Wikipedia*  
- **Author**: Amir Pourmand  
- **Platform**: Kaggle  
- **Year**: 2022  
- **License**: CC0 (Public Domain)  
- **Snapshot Date**: 1400/04/25  
- **Format**: CSV (`title`, `content`, `link`)  
- **Size**: ~4.91 GB  

Dataset link:  
https://www.kaggle.com/datasets/amirpourmand/fa-wikipedia

The raw Wikipedia corpus **is not redistributed** in this repository due to size and licensing best practices. Users must download it directly from Kaggle.

---

## Methodology

### Preprocessing
The pipeline applies a Persian-specific preprocessing workflow:
- HTML tag removal (BeautifulSoup)
- Noise filtering and normalization
- Persian text normalization (Hazm)
- Tokenization and stopword removal (Hazm)

### Named Entity Recognition
- **NER Engine**: Stanza (Persian pretrained model)
- **Entity Types Extracted**:
  - PER (Person)
  - LOC (Location)
  - ORG (Organization)
  - FAC (Facility)
  - PRO (Product)
  - EVENT (Event)

NER is applied in **batch mode** with:
- Configurable batch size
- Logging and progress tracking
- Power-failure recovery via resume checkpoints

### Output Artifacts
The pipeline produces **silver-standard entity inventories**, automatically extracted from the corpus:
- `per_entities`
- `loc_entities`
- `org_entities`
- `fac_entities`
- `pro_entities`
- `event_entities`

These artifacts are intended for **research and analysis**, not as gold-standard annotations.

---

## Generated Resources

The following entity inventories were extracted from the Wikipedia snapshot:

| Entity Type | Approx. Unique Entries |
|------------|------------------------|
| PER        | 765,974 |
| LOC        | 294,747 |
| ORG        | 211,018 |
| FAC        | 97,162 |
| PRO        | 41,088 |
| EVENT     | 39,595 |

Due to size constraints, full artifacts are **not committed directly to GitHub**.

### Artifact Access
- Full entity inventories are distributed via **GitHub Releases**
- Each release includes:
  - Entity files (CSV / Parquet)
  - SHA-256 checksums
  - Extraction metadata (pipeline version, corpus snapshot)

See `artifacts/README.md` for links and integrity verification.

---

## Repository Structure

```
persian-ner-pipeline/
│
├── pipeline.py            # Main NER pipeline script
├── artifacts/             # Artifact metadata + integrity files
│   ├── README.md
│   ├── manifest.json
│   └── checksums_sha256.txt
├── data/
│   └── README.md          # Instructions for obtaining the Kaggle dataset
├── README.md              # This file
├── LICENSE
├── .gitignore
└── .gitattributes
```

Generated outputs (e.g., CSV results, logs) are written to a local `output/` directory,
which is intentionally **gitignored** and not part of the repository.

---

## Reproducibility

To ensure reproducibility:
- All experiments are tied to a **fixed Wikipedia snapshot**
- Pipeline versions are tracked via **Git commit hashes**
- Dependency versions are explicitly documented
- Artifact checksums are provided for verification

This repository follows **research software best practices** expected by Q1 journals.

---

## Installation

```bash
conda create -n persian_ner python=3.10 -y
conda activate persian_ner
pip install pandas beautifulsoup4 hazm stanza
```

Download the Persian Stanza model:
```python
import stanza
stanza.download("fa")
```

---

## Usage

```bash
python pipeline.py
```

The script expects the Wikipedia CSV file to be available at:
```
data/raw_data.csv
```

---

## Limitations

- The extracted entities are **automatically generated** and may contain noise.
- No manual annotation or gold-standard evaluation is provided.
- Wikipedia text includes ambiguity and inconsistencies.

Users are encouraged to apply **manual validation or downstream filtering** when using these resources.

---

## Citation

If you use this pipeline or the derived resources, please cite:

```
@misc{persian_ner_pipeline_2026,
  title        = {Persian NER Pipeline for Wikipedia-based Entity Extraction},
  author       = {Amir Radnia},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/amirradnia99/persian-ner-pipeline}
}
```

And the original dataset:

```
@misc{amir_pourmand_2022,
  title        = {Farsi Wikipedia},
  author       = {Pourmand, Amir},
  year         = {2022},
  publisher    = {Kaggle},
  doi          = {10.34740/KAGGLE/DSV/3949764}
}
```

---

## License
MIT License © Amir Radnia
