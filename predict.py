import os
import joblib
import logging
import argparse
import glob
from utils.logger import setup_logger

def predict(text_file, model_folder='models', output_folder='output', logger_name='prediction'):
    logger = logging.getLogger(logger_name)
    try:
        logger.info(f"Predicting text from {text_file} using Naive Bayes model")
        
        try:
            with open(text_file, 'r', encoding='utf-8', errors='replace') as file:
                text = file.read()
        except Exception as e:
            logger.error(f"Error reading text file {text_file}: {e}")
            return None
        
        model_path = os.path.join(model_folder, 'naive_bayes_model.pkl')
        vectorizer_path = os.path.join(model_folder, 'naive_bayes_vectorizer.pkl')
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)
        
        result = 'spam' if prediction[0] == 1 else 'ham'
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        output_file_name = f'output_prediction_{os.path.basename(text_file)}'
        if not output_file_name.endswith('.txt'):
            output_file_name += '.txt'
        
        output_path = os.path.join(output_folder, output_file_name)
        with open(output_path, 'w') as output_file:
            output_file.write(result)
        
        logger.info(f"Prediction for {text_file} saved to {output_path}")
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Predict using Spam Filter Model')
    parser.add_argument('--input_directory', required=True, help='Path to the input directory containing text files')
    parser.add_argument('--input_files', required=True, help='Names of the input text files (use quotes and * for all files in directory)')
    parser.add_argument('--model_folder', default='models', help='Path to the folder containing the trained model')
    parser.add_argument('--output_folder', default='output', help='Path to the folder where the output will be saved')
    args = parser.parse_args()

    input_directory = args.input_directory
    input_files_pattern = args.input_files
    model_folder = args.model_folder
    output_folder = args.output_folder

    logger = setup_logger('prediction.log')

    # Check if input directory exists
    if not os.path.exists(input_directory):
        logger.error(f"Input directory {input_directory} does not exist.")
        return

    # Get input files
    input_files = glob.glob(os.path.join(input_directory, input_files_pattern))
    if not input_files:
        logger.error(f"No input files found in {input_directory}.")
        return

    try:
        logger.info("Starting spam filter prediction program")
        
        for text_file in input_files:
            result = predict(text_file, model_folder, output_folder, logger_name='prediction')
            if result:
                logger.info(f"Prediction result for {text_file}: {result}")
        
    except Exception as e:
        logger.error(f"Error in main program: {e}")

if __name__ == '__main__':
    main()