# Data Directory

This directory documents the **input data contract** for the Persian NER Pipeline (PW-NER).

⚠️ **No raw data is tracked in this repository.**  
This directory exists solely to document **how to obtain and prepare the input corpus** required to reproduce the released artifacts.

---

## Required Input File

After downloading the corpus, the expected file location is:

```
data/raw_data.csv
```

This file **must not** be committed to the repository.

---

## Source Corpus

- **Corpus:** Farsi Wikipedia  
- **Provider:** Amir Pourmand  
- **Platform:** Kaggle  
- **License:** CC0 (Public Domain)  
- **Snapshot date:** **1400/04/25**

Download link:
https://www.kaggle.com/datasets/amirpourmand/fa-wikipedia

---

## Expected CSV Schema

The CSV file must contain the following columns:

| Column name | Required | Description |
|------------|----------|-------------|
| `title`    | Yes      | Wikipedia article title |
| `content`  | Yes      | Raw article text (may include HTML markup) |
| `link`     | Optional | Article URL |

All columns must be UTF-8 encoded.

---

## Notes on Reproducibility

- The **exact snapshot date matters**. Using a different Wikipedia dump will change entity distributions.
- HTML markup is expected; it is removed internally by the pipeline.
- No tokenization or stopword removal should be applied before running the pipeline.

For full reproduction instructions, see:
```
docs/REPRODUCIBILITY.md
```

---

## Why the Data Is Not Included

The raw Wikipedia corpus is:

- very large
- publicly available elsewhere
- licensed independently (CC0)

Redistributing it here would be redundant and unnecessary.

This repository instead provides:
- the **exact extraction code**
- **silver-standard derived inventories**
- full **provenance and integrity metadata**
