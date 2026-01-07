# Named Entity Recognition (NER) on Persian Wikipedia Content
# ------------------------------------------------------------
# This script performs preprocessing and Named Entity Recognition on Persian text using Hazm and Stanza.
# Features:
# - Text normalization with Hazm
# - Batch processing with configurable size
# - Robust fault tolerance with reliable resume logic
# - Progress tracking and logging
# - Full document provenance with Wikipedia links
# - Clean JSON storage of extracted entities

import os
import re
import json
import pandas as pd
import logging
from bs4 import BeautifulSoup
from hazm import Normalizer
import stanza

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
raw_file_path = 'data/raw_data.csv'                        # Input dataset
preprocessed_path = 'output/preprocessed_data.csv'         # After cleaning & normalization
processed_output_path = 'output/processed_data.csv'        # Final output with NER
progress_path = 'output/progress.txt'                      # Save progress for fault recovery

# Processing settings
batch_size = 1000
print_interval = 100
REPORT_STATS = False  # Set to True for detailed batch statistics

# Ensure output directories exist
os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)

# ------------------------------------------------------------
# 2. Load and Prepare the Dataset
# ------------------------------------------------------------
df = pd.read_csv(raw_file_path)
df['title'] = df['title'].astype(str)
df['content'] = df['content'].astype(str)

# Handle link column gracefully for provenance
if 'link' in df.columns:
    df['link'] = df['link'].astype(str)
    logging.info("✅ Link column found and preserved for provenance")
else:
    df['link'] = ''  # Create empty link column if missing
    logging.warning("⚠️  Link column not found - creating empty column")

# ------------------------------------------------------------
# 3. Remove HTML Tags
# ------------------------------------------------------------
def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

df['cleaned_title'] = df['title'].apply(remove_html_tags)
df['cleaned_content'] = df['content'].apply(remove_html_tags)

# ------------------------------------------------------------
# 4. Normalize Text with Hazm (preserves structure for NER)
# ------------------------------------------------------------
normalizer = Normalizer()

def normalize_persian_text(text):
    """Normalize Persian text using Hazm while preserving structure for NER."""
    # Apply Hazm normalization (handles Persian-specific issues)
    text = normalizer.normalize(text)
    
    # Clean up excessive whitespace but preserve sentence boundaries
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Keep all characters including digits, Latin, and punctuation
    # This preserves entities like dates, chemical names, and mixed-script terms
    return text

df['normalized_title'] = df['cleaned_title'].apply(normalize_persian_text)
df['normalized_content'] = df['cleaned_content'].apply(normalize_persian_text)

# Save preprocessed file (for debugging or reuse)
df.to_csv(preprocessed_path, index=False)
logging.info(f"✅ Preprocessed data saved to: {preprocessed_path}")

# ------------------------------------------------------------
# 5. Initialize Stanza NER Pipeline
# ------------------------------------------------------------
# stanza.download('fa')  # Uncomment this line manually if not already downloaded

# Initialize with explicit settings for reproducibility
try:
    nlp = stanza.Pipeline(
        lang='fa', 
        processors='tokenize,ner',
        use_gpu=False,           # Ensure CPU-only for consistency
        verbose=False            # Suppress verbose output
    )
    logging.info("✅ Stanza NER pipeline initialized")
except Exception as e:
    logging.error(f"❌ Failed to initialize Stanza pipeline: {e}")
    raise

# ------------------------------------------------------------
# 6. Named Entity Recognition Function
# ------------------------------------------------------------
def recognize_entities(text):
    """Extract named entities from text and return as clean JSON-serializable list.
    
    Note: Character offsets (start_char, end_char) refer to positions in the 
    normalized text after Hazm processing, not the original raw text.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    try:
        doc = nlp(text)
        entities = []
        for sent in doc.sentences:
            for ent in sent.ents:
                entities.append({
                    'text': ent.text,
                    'type': ent.type,
                    'start_char': ent.start_char,  # Offset in normalized text
                    'end_char': ent.end_char       # Offset in normalized text
                })
        return entities
    except Exception as e:
        logging.warning(f"⚠️  NER failed for text (length: {len(text)}): {e}")
        return []

# ------------------------------------------------------------
# 7. Determine Resume Point and Header Logic
# ------------------------------------------------------------
# Single source of truth: progress.txt
start_index = 0
if os.path.exists(progress_path):
    try:
        with open(progress_path, 'r') as f:
            saved_index = int(f.read().strip())
            start_index = saved_index
            logging.info(f"📁 Resuming from row {start_index} (from progress.txt)")
    except (ValueError, IOError) as e:
        logging.warning(f"⚠️  Could not read progress.txt: {e}. Starting from 0.")

# Determine if we need to write header
# Header is needed if: starting from 0 AND (file doesn't exist OR is empty)
need_header = (start_index == 0) and (
    not os.path.exists(processed_output_path) or 
    os.path.getsize(processed_output_path) == 0
)

if start_index == 0:
    if os.path.exists(processed_output_path) and os.path.getsize(processed_output_path) > 0:
        logging.info("🔄 Output file exists but appears incomplete. Will append without header.")
        need_header = False

logging.info(f"🚀 Starting NER processing from row {start_index} of {len(df)}")
if need_header:
    logging.info("📝 Will write header to output file")

# ------------------------------------------------------------
# 8. Run Batch Processing Loop
# ------------------------------------------------------------
with open(processed_output_path, 'a', encoding='utf-8') as f:
    for i in range(start_index, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))

        for j in range(i, batch_end):
            try:
                # Extract entities from normalized text (natural text structure)
                title_entities = recognize_entities(df.at[j, 'normalized_title'])
                content_entities = recognize_entities(df.at[j, 'normalized_content'])
                
                # Store as JSON strings for clean serialization
                df.at[j, 'entities_title'] = json.dumps(title_entities, ensure_ascii=False)
                df.at[j, 'entities_content'] = json.dumps(content_entities, ensure_ascii=False)
                
                if (j + 1) % print_interval == 0:
                    logging.info(f"🧠 Processed {j + 1} rows")
                    
            except Exception as e:
                logging.error(f"❌ Error processing row {j}: {e}")
                # Store empty JSON arrays for failed rows
                df.at[j, 'entities_title'] = json.dumps([], ensure_ascii=False)
                df.at[j, 'entities_content'] = json.dumps([], ensure_ascii=False)

        # Prepare batch for output - only essential columns for entity inventory
        # Includes provenance (title, link) and extracted entities
        batch_df = df.iloc[i:batch_end][[
            'title', 
            'link', 
            'entities_title', 
            'entities_content'
        ]].copy()
        
        # Append batch to file with correct header logic
        batch_df.to_csv(f, header=need_header, index=False)
        need_header = False  # Only write header once
        
        # Update progress - single source of truth
        with open(progress_path, 'w') as p:
            p.write(str(batch_end))
        
        logging.info(f"💾 Saved batch {i//batch_size + 1}: rows {i} to {batch_end-1}")
        
        # Optional batch statistics (disabled by default for performance)
        if REPORT_STATS:
            batch_entity_count = 0
            for j in range(i, batch_end):
                try:
                    batch_entity_count += len(json.loads(df.at[j, 'entities_title']))
                    batch_entity_count += len(json.loads(df.at[j, 'entities_content']))
                except:
                    pass
            logging.info(f"📊 Batch {i//batch_size + 1}: extracted {batch_entity_count} entities")

logging.info("✅ NER Processing Completed Successfully.")

# ------------------------------------------------------------
# 9. Final Summary (always show basic stats)
# ------------------------------------------------------------
if os.path.exists(processed_output_path):
    try:
        final_df = pd.read_csv(processed_output_path)
        total_rows = len(final_df)
        
        # Count total entities efficiently
        total_title_entities = 0
        total_content_entities = 0
        
        for ents_title, ents_content in zip(final_df['entities_title'], final_df['entities_content']):
            if isinstance(ents_title, str) and ents_title:
                try:
                    total_title_entities += len(json.loads(ents_title))
                except:
                    pass
            if isinstance(ents_content, str) and ents_content:
                try:
                    total_content_entities += len(json.loads(ents_content))
                except:
                    pass
        
        total_entities = total_title_entities + total_content_entities
        
        logging.info("=" * 50)
        logging.info("📈 PROCESSING SUMMARY")
        logging.info("=" * 50)
        logging.info(f"Total documents processed: {total_rows}")
        logging.info(f"Entities in titles: {total_title_entities}")
        logging.info(f"Entities in content: {total_content_entities}")
        logging.info(f"Total entities extracted: {total_entities}")
        logging.info(f"Average entities per document: {total_entities/total_rows:.2f}")
        logging.info(f"Output file size: {os.path.getsize(processed_output_path) / (1024**2):.2f} MB")
        logging.info(f"Output file: {processed_output_path}")
        logging.info("=" * 50)
        
    except Exception as e:
        logging.error(f"❌ Could not generate final statistics: {e}")