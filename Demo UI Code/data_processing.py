import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_clean_data(file_path):
    logger.info(f"Loading data from {file_path}")
    try:
        columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party',
                   'barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire', 'context']
        df = pd.read_csv(file_path, sep='\t', names=columns, header=None)
        df = df[['label', 'statement']]
        df['label'] = df['label'].map({
            'true': 1, 'mostly-true': 1, 'half-true': 0, 'barely-true': 0, 'false': -1, 'pants-fire': -1
        })
        return df.dropna()  # Drop rows with NaN labels
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def save_processed_data(df, output_path):
    if df is not None and not df.empty:
        logger.info(f"Saving to {output_path}")
        df.to_csv(output_path, index=False)
    else:
        logger.error("No valid data to save.")

def process_data(splits, raw_data_dir="/Users/mattlaing/Desktop/TruthSeeker/data/raw", processed_data_dir="/Users/mattlaing/Desktop/TruthSeeker/data/processed"):
    os.makedirs(processed_data_dir, exist_ok=True)
    for split in splits:
        input_path = f"{raw_data_dir}/{split}.tsv"
        output_path = f"{processed_data_dir}/{split}.csv"
        df = load_and_clean_data(input_path)
        save_processed_data(df, output_path)

if __name__ == "__main__":
    process_data(['train', 'test', 'valid'])
