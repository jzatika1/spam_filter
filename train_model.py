import os
import sys
import argparse
import glob
import logging
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from utils.data_loader import load_and_preprocess_data
from utils.logger import setup_logger
from utils.evaluation import evaluate_model_performance, print_data_information

def train_naive_bayes(data, model_folder='models', hyperparameter_optimization=False, logger=None):
    try:
        logger.info("Training Naive Bayes model")

        X = data['message']
        y = data['label']

        # CountVectorizer
        vectorizer = CountVectorizer()
        X_vectorized = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.20, random_state=0)

        # Print data information
        print_data_information(X_train, X_test, y_train, y_test, logger)

        if hyperparameter_optimization:
            # Define the parameter grid
            param_grid = {
                'alpha': [0.01, 0.1, 0.5, 1.0, 1.5, 2.0],
                'fit_prior': [True, False]
            }

            # Initialize GridSearchCV
            grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Get the best model
            model = grid_search.best_estimator_
            logger.info(f"Best parameters found: {grid_search.best_params_}")

            # Evaluate the best model
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            evaluate_model_performance(y_test, y_pred, y_pred_proba, logger)

            return model, vectorizer  # Do not save the model during hyperparameter optimization
        else:
            # Train the Naive Bayes model
            model = MultinomialNB(alpha=0.01, fit_prior=True)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            evaluate_model_performance(y_test, y_pred, y_pred_proba, logger)

            if not os.path.exists(model_folder):
                os.makedirs(model_folder)

            model_path = os.path.join(model_folder, 'naive_bayes_model.pkl')
            vectorizer_path = os.path.join(model_folder, 'naive_bayes_vectorizer.pkl')

            joblib.dump(model, model_path)
            joblib.dump(vectorizer, vectorizer_path)

            logger.info(f"Naive Bayes model and vectorizer saved to {model_folder}")
            return model, vectorizer
    except Exception as e:
        logger.error(f"Error training Naive Bayes model: {e}")
        raise

def train_model(training_data_files, model_folder='models', hyperparameter_optimization=False, logger=None):
    try:
        logger.info("Starting training for Naive Bayes model")

        data = load_and_preprocess_data(training_data_files, logger_name='training')
        model, vectorizer = train_naive_bayes(data, model_folder, hyperparameter_optimization, logger)

        logger.info("Training completed for Naive Bayes model")
        return model, vectorizer
    except Exception as e:
        logger.error(f"Error in training model: {e}")
        print(f"Error in training model: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Train Spam Filter Model')
    parser.add_argument('--model', choices=['naive_bayes'], required=True, help='Specify the model to use')
    parser.add_argument('--training_data_directory', required=True, help='Path to the folder containing training data files')
    parser.add_argument('--training_data_files', required=True, help='Names of the training data files (use quotes and * for all files in directory)')
    parser.add_argument('--hyperparameter_optimization', action='store_true', help='Perform hyperparameter optimization')
    args = parser.parse_args()

    training_data_directory = args.training_data_directory
    training_data_files_pattern = args.training_data_files
    perform_hyperparameter_optimization = args.hyperparameter_optimization

    logger_name = 'hyperparameter_optimization.log' if perform_hyperparameter_optimization else 'training.log'
    logger = setup_logger(logger_name)

    # Check if training data directory exists
    if not os.path.exists(training_data_directory):
        logger.error(f"Training data directory {training_data_directory} does not exist.")
        print(f"Error: Training data directory {training_data_directory} does not exist.")
        return

    # Get training data files
    training_data_files = glob.glob(os.path.join(training_data_directory, training_data_files_pattern))
    if not training_data_files:
        logger.error(f"No training data files found in {training_data_directory}.")
        print(f"Error: No training data files found in {training_data_directory}.")
        return

    try:
        logger.info("Starting spam filter training program")

        data = load_and_preprocess_data(training_data_files, logger_name=logger_name)

        if perform_hyperparameter_optimization:
            logger.info("Performing hyperparameter optimization")
            model, vectorizer = train_naive_bayes(data, model_folder='models', hyperparameter_optimization=True, logger=logger)
        else:
            logger.info("Training model without hyperparameter optimization")
            model, vectorizer = train_model(training_data_files, model_folder='models', logger=logger)

        logger.info("Spam filter training program completed successfully")

    except Exception as e:
        logger.error(f"Error in main program: {e}")
        print(f"Error in main program: {e}")

if __name__ == '__main__':
    main()