# Named Entity Recognition (NER) on Persian Wikipedia Content
# ------------------------------------------------------------
# This script performs preprocessing and Named Entity Recognition on Persian text using Hazm and Stanza.
# Features:
# - Complete text preprocessing pipeline
# - Batch processing with configurable size
# - Power-failure recovery system
# - Progress tracking and logging

import os
import re
import pandas as pd
import logging
from bs4 import BeautifulSoup
from hazm import Normalizer, word_tokenize, stopwords_list
import stanza

# ------------------------------------------------------------
# 1. Setup Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------
# 2. User-Defined File Paths
# ------------------------------------------------------------
raw_file_path = 'data/raw_data.csv'                        # Input dataset
preprocessed_path = 'output/preprocessed_data.csv'         # After cleaning & normalization
processed_output_path = 'output/processed_data.csv'        # Final output with NER
progress_path = 'output/progress.txt'                      # Save progress for fault recovery

# Ensure output directories exist
os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)

# ------------------------------------------------------------
# 3. Load and Prepare the Dataset
# ------------------------------------------------------------
df = pd.read_csv(raw_file_path)
df['title'] = df['title'].astype(str)
df['content'] = df['content'].astype(str)
df = df.drop(columns=['link'], errors='ignore')  # Drop if exists

# ------------------------------------------------------------
# 4. Remove HTML Tags
# ------------------------------------------------------------
def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

df['cleaned_title'] = df['title'].apply(remove_html_tags)
df['cleaned_content'] = df['content'].apply(remove_html_tags)

# ------------------------------------------------------------
# 5. Remove Non-Persian Characters
# ------------------------------------------------------------
def remove_non_persian_characters(text):
    return re.sub(r'[^\u0600-\u06FF\s]', '', text)

df['cleaned_title'] = df['cleaned_title'].apply(remove_non_persian_characters)
df['cleaned_content'] = df['cleaned_content'].apply(remove_non_persian_characters)

# ------------------------------------------------------------
# 6. Normalize Text (Spaces and Punctuation)
# ------------------------------------------------------------
def normalize_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['cleaned_title'] = df['cleaned_title'].apply(normalize_text)
df['cleaned_content'] = df['cleaned_content'].apply(normalize_text)

# ------------------------------------------------------------
# 7. Tokenization and Stopword Removal with Hazm
# ------------------------------------------------------------
normalizer = Normalizer()
stop_words = set(stopwords_list())

def preprocess_text(text):
    text = normalizer.normalize(text)
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

df['preprocessed_title'] = df['cleaned_title'].apply(preprocess_text)
df['preprocessed_content'] = df['cleaned_content'].apply(preprocess_text)

# Save preprocessed file
df.to_csv(preprocessed_path, index=False)
logging.info(f"✅ Preprocessed data saved to: {preprocessed_path}")

# ------------------------------------------------------------
# 8. Reload Preprocessed Data for NER
# ------------------------------------------------------------
df = pd.read_csv(preprocessed_path)

# stanza.download('fa')  # Uncomment this line manually if not already downloaded
nlp = stanza.Pipeline(lang='fa', processors='tokenize,ner')

# ------------------------------------------------------------
# 9. Named Entity Recognition Function
# ------------------------------------------------------------
def recognize_entities(text):
    if not isinstance(text, str):
        return []
    doc = nlp(text)
    return [(ent.text, ent.type) for sent in doc.sentences for ent in sent.ents]

# ------------------------------------------------------------
# 10. Batch NER Processing with Fault Tolerance
# ------------------------------------------------------------
batch_size = 1000
print_interval = 100

# Add empty columns only if file doesn't exist
if not os.path.exists(processed_output_path):
    df['entities_title'] = pd.NA
    df['entities_content'] = pd.NA

# Determine resume point
start_index = 0
if os.path.exists(processed_output_path):
    processed_df = pd.read_csv(processed_output_path)
    start_index = len(processed_df)
    logging.info(f"📁 Resuming from row {start_index} (processed_data.csv exists)")

if os.path.exists(progress_path):
    with open(progress_path, 'r') as f:
        saved_index = int(f.read().strip())
        start_index = max(start_index, saved_index)
    logging.info(f"📁 Resuming from row {start_index} (progress.txt exists)")

# ------------------------------------------------------------
# 11. Run Batch Processing Loop
# ------------------------------------------------------------
with open(processed_output_path, 'a', encoding='utf-8') as f:
    for i in range(start_index, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))

        for j in range(i, batch_end):
            try:
                df.at[j, 'entities_title'] = str(recognize_entities(df.at[j, 'cleaned_title']))
                df.at[j, 'entities_content'] = str(recognize_entities(df.at[j, 'cleaned_content']))
                if (j + 1) % print_interval == 0:
                    logging.info(f"🧠 Processed {j + 1} rows")
            except Exception as e:
                logging.error(f"❌ Error processing row {j}: {e}")

        # Save only relevant columns
        df.iloc[i:batch_end][[
            'title', 'content', 'entities_title', 'entities_content'
        ]].to_csv(f, header=(i == start_index), index=False, mode='a')

        # Save progress
        with open(progress_path, 'w') as p:
            p.write(str(batch_end))

        logging.info(f"💾 Saved up to row {batch_end}")

logging.info("✅ NER Processing Completed Successfully.")
