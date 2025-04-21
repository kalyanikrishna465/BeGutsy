import numpy as np
import logging
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

def classify_samples(health_indices, threshold):
    """
    Classify samples based on the combined health index.
    If combined_index <= threshold then sample is considered 'Healthy' (1),
    otherwise 'Diseased' (0). (Adjust rule as needed based on index definition.)
    Returns an array of predicted labels.
    """
    try:
        predictions = np.array([1 if idx <= threshold else 0 for idx in health_indices])
        logging.info(f"Classification completed on {len(health_indices)} samples with threshold {threshold}.")
        return predictions
    except Exception as e:
        logging.error(f"Error in classify_samples: {e}")
        raise e

def evaluate_classification(true_labels, predicted_labels):
    """
    Compute balanced accuracy and AUC score.
    """
    try:
        bal_acc = balanced_accuracy_score(true_labels, predicted_labels)
        try:
            auc = roc_auc_score(true_labels, predicted_labels)
        except Exception as e:
            logging.warning(f"Could not compute AUC: {e}")
            auc = np.nan
        logging.info(f"Classification evaluation: Balanced Accuracy = {bal_acc}, AUC = {auc}")
        return {"balanced_accuracy": bal_acc, "auc": auc}
    except Exception as e:
        logging.error(f"Error in evaluate_classification: {e}")
        raise e
