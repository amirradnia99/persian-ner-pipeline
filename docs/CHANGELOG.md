# Changelog — Persian NER Pipeline (PW-NER)

All notable changes to this repository and its released artifacts are documented in this file.

This project follows **semantic versioning** for releases related to published datasets and reproducibility-critical changes.

---

## [Unreleased]

### Added
- Streaming (chunked) CSV ingestion in `pipeline.py` to avoid loading the full corpus in memory
- End-to-end dataset export from `pipeline.py` via `--export_inventories` (writes `inventories/*.xlsx` or `.csv`)
- Deterministic manifest option (`--no_timestamp`) for `pipeline.py`
- Deterministic QC report option (`--no_timestamp`) for `qc.py`
- Deterministic checksum generator script (`scripts/make_checksums.py`) and manifest linkage
- Minimal deterministic CI smoke test (no Stanza model download) + `python -m compileall .`
- Optional `requirements-dev.txt` for linting/testing

### Fixed
- Explicit Hazm normalizer configuration recorded in manifests (stabilizes preprocessing contract)
- Deterministic JSON output ordering for QC reports (`sort_keys=True`) and stable rule ordering
- More robust Excel ingestion in QC (force text conversion; avoid NA coercion)

### Notes
- If you re-run extraction with the updated pipeline/QC, output files may differ from older runs
  unless you keep the same corpus snapshot, dependencies, and flags. Dataset releases must be
  versioned when counts/contents change.

---

## [v1.0.1-silver] — 2026-01-07

### Added
- Minor documentation refinements in `README.md`
- Explicit reference to latest silver-standard release asset
- Clarified artifact verification instructions

### Fixed
- README alignment with released entity counts
- Minor wording inconsistencies related to corpus snapshot description

### Notes
- No changes to extraction logic or QC rules
- Dataset content identical to `v1.0.0-silver`

---

## [v1.0.0-silver] — 2026-01-06

### Added
- Initial public release of **PW-NER silver-standard Persian NER inventories**
- Entity classes released:
  - PER, LOC, ORG, FAC, PRO, EVENT
- Full artifact provenance via `artifacts/manifest.json`
- SHA-256 integrity verification (`artifacts/checksums_sha256.txt`)
- Machine-readable citation metadata (`CITATION.cff`)

### Pipeline
- Persian text normalization using **Hazm**
- Named Entity Recognition using **Stanza (fa)**
- CPU-only deterministic inference
- Fault-tolerant batch processing

### Dataset
- Source corpus: Farsi Wikipedia (Kaggle, CC0)
- Snapshot date: **1400/04/25**
- Raw corpus not redistributed

### Limitations
- Automatically extracted (silver-standard)
- Contains noise from Wikipedia templates, lists, and formatting artifacts

---

## Versioning Policy

- **MAJOR**: Breaking changes to dataset schema or entity definitions
- **MINOR**: Additions or improvements that do not invalidate existing releases
- **PATCH**: Documentation fixes or non-functional changes

---

## Citation Impact

If a change alters:
- entity counts
- extraction logic
- normalization rules
- QC thresholds

then a **new dataset version must be released and cited separately**.
