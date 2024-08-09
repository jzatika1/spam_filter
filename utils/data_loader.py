import pandas as pd
import logging
from multiprocessing import Pool, cpu_count

def load_file(file_path, logger_name):
    logger = logging.getLogger(logger_name)
    try:
        if 'SMSSpamCollection' in file_path:
            data = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
        else:
            data = pd.read_csv(file_path)
        
        # Normalize column names
        if 'labels' in data.columns:
            data.rename(columns={'labels': 'label'}, inplace=True)
        if 'body' in data.columns:
            data.rename(columns={'body': 'message'}, inplace=True)
        if 'Body' in data.columns:
            data.rename(columns={'Body': 'message'}, inplace=True)
        if 'Label' in data.columns:
            data.rename(columns={'Label': 'label'}, inplace=True)
        if 'text_combined' in data.columns:
            data.rename(columns={'text_combined': 'message'}, inplace=True)
        if 'text' in data.columns:
            data.rename(columns={'text': 'message'}, inplace=True)

        # Ensure we have the columns we need
        if 'label' in data.columns and 'message' in data.columns:
            # Print the count of spam and ham messages
            if data['label'].dtype == object:
                data['label'] = data['label'].map({'ham': 0, 'spam': 1})
            spam_count = sum(data['label'] == 1)
            ham_count = sum(data['label'] == 0)
            logger.info(f"File {file_path} - Spam: {spam_count}, Ham: {ham_count}")
            return data[['label', 'message']]
        else:
            logger.warning(f"File {file_path} does not contain the required columns.")
            return pd.DataFrame(columns=['label', 'message'])

    except pd.errors.ParserError as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return pd.DataFrame(columns=['label', 'message'])

def load_and_preprocess_data(file_paths, logger_name):
    logger = logging.getLogger(logger_name)
    try:
        logger.info(f"Loading data from {file_paths}")

        with Pool(processes=cpu_count()) as pool:
            data_frames = pool.starmap(load_file, [(file_path, logger_name) for file_path in file_paths])

        data = pd.concat(data_frames, ignore_index=True)

        # Drop rows with NaN values in 'message'
        data.dropna(subset=['message'], inplace=True)

        # Drop duplicate rows based on 'message'
        data.drop_duplicates(subset=['message'], inplace=True)

        # Map label values if they are not already numerical
        if data['label'].dtype == object:
            data['label'] = data['label'].map({'ham': 0, 'spam': 1})
            data['label'] = data['label'].fillna(0).astype(int)

        # Remove leading and trailing spaces
        data['message'] = data['message'].str.strip()

        logger.info("Data loaded, preprocessed, and duplicates removed")
        return data
    except Exception as e:
        logger.error(f"Error loading or preprocessing data: {e}")
        raise