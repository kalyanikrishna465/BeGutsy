import numpy as np
import logging
from pca_module import compute_T2_index, compute_Q_index, compute_control_limits, compute_combined_index

def evaluate_sample_control_charts(sample, pca_model, T2_limit, Q_limit):
    """
    Given a sample (as 1D np.array of preprocessed features) and PCA model,
    compute T2, Q and combined index.
    Returns a dictionary with the values.
    """
    try:
        T2 = compute_T2_index(sample, pca_model)
        Q = compute_Q_index(sample, pca_model)
        combined = compute_combined_index(T2, Q, T2_limit, Q_limit)
        logging.debug(f"Sample evaluated: T2 = {T2}, Q = {Q}, Combined index = {combined}")
        return {"T2": T2, "Q": Q, "combined_index": combined}
    except Exception as e:
        logging.error(f"Error in evaluate_sample_control_charts: {e}")
        raise e
