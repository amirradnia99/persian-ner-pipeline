# Data Directory

This directory is reserved for **input data files** required to run the Persian NER pipeline.

## Required Dataset

### Farsi Wikipedia (Kaggle)

- **Name**: Farsi Wikipedia
- **Author**: Amir Pourmand
- **Year**: 2022
- **License**: CC0 (Public Domain)
- **Platform**: Kaggle
- **Link**: https://www.kaggle.com/datasets/amirpourmand/fa-wikipedia

This dataset contains all Persian Wikipedia articles in a single CSV file.

### Expected File Format

After downloading from Kaggle, place the file here and rename it as:

```
data/raw_data.csv
```

The CSV file must contain at least the following columns:

- `title`   – Wikipedia article title  
- `content` – Article text  
- `link`    – Permanent Wikipedia URL  

No data files are tracked by Git to avoid storing large or licensed content.

## Notes on Reproducibility

- The experiments and artifacts released with this repository were generated using the
  Kaggle snapshot dated **1400/04/25**.
- Using a different snapshot may result in different entity counts.
- The pipeline assumes UTF-8 encoded Persian text.

## Small-Scale Testing (Optional)

For testing purposes, users may create a small subset of the dataset
(e.g., first 100 rows) and place it in this directory as `raw_data.csv`.

## Important

⚠️ **Do not commit large data files to this repository.**  
All derived artifacts are distributed via GitHub Releases with integrity checks.
