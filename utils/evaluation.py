import logging
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, roc_auc_score
import numpy as np

def print_data_information(X_train, X_test, y_train, y_test, logger):
    try:
        train_size = len(y_train)
        test_size = len(y_test)
        total_size = train_size + test_size
        train_percentage = (train_size / total_size) * 100
        test_percentage = (test_size / total_size) * 100
        
        spam_train = sum(y_train)
        ham_train = train_size - spam_train
        spam_test = sum(y_test)
        ham_test = test_size - spam_test
        
        logger.info(f"Data Information:")
        logger.info(f"Total data size: {total_size}")
        logger.info(f"Train set size: {train_size} ({train_percentage:.2f}%)")
        logger.info(f"Test set size: {test_size} ({test_percentage:.2f}%)")
        logger.info(f"Train set - Spam: {spam_train}, Ham: {ham_train}")
        logger.info(f"Test set - Spam: {spam_test}, Ham: {ham_test}")
    except Exception as e:
        logger.error(f"Error printing data information: {e}")

def evaluate_model_performance(y_test, y_pred, y_pred_proba=None, logger=None):
    try:
        if y_pred_proba is not None and y_pred_proba.ndim > 1:
            preds_classes = np.argmax(y_pred_proba, axis=1)
        else:
            preds_classes = y_pred.round()

        accuracy = accuracy_score(y_test, preds_classes)
        precision = precision_score(y_test, preds_classes, average='binary', zero_division=1)
        recall = recall_score(y_test, preds_classes, average='binary', zero_division=1)
        
        # Manually calculate F1 Score
        manual_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        # Calculate ROC-AUC score if probabilities are provided
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1]) if y_pred_proba is not None and y_pred_proba.ndim > 1 else None

        conf_matrix = confusion_matrix(y_test, preds_classes)
        tn, fp, fn, tp = conf_matrix.ravel()

        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': manual_f1,
            'ROC-AUC Score': roc_auc,
        }

        if logger:
            logger.info(f"Model Performance:")
            for metric, value in metrics.items():
                if value is not None:
                    logger.info(f"{metric}: {value}")
            logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        else:
            print(f"Model Performance:")
            for metric, value in metrics.items():
                if value is not None:
                    print(f"{metric}: {value}")
            print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        return metrics
    except Exception as e:
        if logger:
            logger.error(f"Error evaluating model performance: {e}")
        else:
            print(f"Error evaluating model performance: {e}")
        return None