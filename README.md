# Persian NER Pipeline

A comprehensive NLP pipeline designed for performing Named Entity Recognition (NER) on Persian Wikipedia content. This repository provides a robust end-to-end solution that includes data cleaning, normalization, tokenization, stopword removal, and NER with the Stanza library, all optimized for Persian text.

---

## Features

- **Complete preprocessing pipeline:**  
  - HTML tag removal  
  - Non-Persian character filtering  
  - Text normalization (whitespace, punctuation)  
  - Tokenization and stopword removal using Hazm  

- **Named Entity Recognition:**  
  - Persian language NER via Stanza’s pretrained models  

- **Batch Processing with Fault Tolerance:**  
  - Configurable batch size for scalable processing  
  - Progress saving and resume capability to handle interruptions  

- **Logging and Progress Tracking:**  
  - Detailed logging for monitoring the pipeline's status  
  - Save and load progress to prevent data loss during long runs  

---

## Requirements

- Python 3.10 or older (recommended)  
- Anaconda or Miniconda environment (recommended for easier package management)

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/persian-ner-pipeline.git
   cd persian-ner-pipeline

2. Create and activate a new conda environment with Python 3.10:
   conda create -n persian_ner python=3.10 -y
   conda activate persian_ner

3. Install required Python packages using pip inside the activated environment:
   pip install pandas beautifulsoup4 hazm stanza

4. Download the Stanza Persian model (if not already installed):
   import stanza
   stanza.download('fa')

## Usage

1. **Prepare your raw dataset** as a CSV file with at least the following columns:
   - `title`
   - `content`

2. **Update the file paths** in `persian_ner_pipeline.py` if needed:

   ```python
   raw_file_path = 'data/raw_data.csv'
   preprocessed_path = 'output/preprocessed_data.csv'
   processed_output_path = 'output/processed_data.csv'
   progress_path = 'output/progress.txt'

## Dependencies

- pandas  
- beautifulsoup4  
- hazm  
- stanza  
- logging (standard library)

## Notes
   This pipeline is tailored for Persian Wikipedia content but can be adapted to other Persian text corpora.

   Adjust batch size and print intervals in the script for optimal performance on your machine.

   Make sure to have enough memory and CPU power for processing large datasets.

   Recommended to use Python 3.10 or older due to compatibility with some dependencies.

## License

MIT License © Amir Radnia

## Contact

For issues or questions, feel free to open an issue or contact me at amir.radnia99@gmail.com
