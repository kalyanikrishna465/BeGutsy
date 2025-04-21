import numpy as np
import pandas as pd
import logging
from scipy.stats import zscore

def log_transform(df, threshold=1, sigma=1e-6):
    """
    Apply logarithmic transformation.
    If value <= threshold, apply log2(2*x + sigma), else leave as is.
    """
    def transform(x):
        if x <= threshold:
            return np.log2(2 * x + sigma)
        else:
            return x
    try:
        transformed = df.map(transform)
        logging.info("Applied log transformation to data.")
        return transformed
    except Exception as e:
        logging.error(f"Error in log_transform: {e}")
        raise e

def normalize_data(df):
    """
    Apply z-score normalization column-wise.
    """
    try:
        normalized = df.apply(zscore)
        logging.info("Applied z-score normalization to data.")
        return normalized
    except Exception as e:
        logging.error(f"Error in normalize_data: {e}")
        raise e

def preprocess_features(merged_data, feature_columns):
    """
    Preprocess the selected feature columns:
    - Log-transform
    - Normalize
    Returns preprocessed dataframe.
    """
    try:
        features = merged_data[feature_columns].copy()
        features_log = log_transform(features)
        features_norm = normalize_data(features_log)
        logging.info("Preprocessed features (log transform and normalization).")
        return features_norm
    except Exception as e:
        logging.error(f"Error in preprocess_features: {e}")
        raise e

def get_feature_columns(merged_data):
    """
    Return list of feature columns (all columns except metadata columns).
    Assumes metadata columns include: Diagnosis, Project, GMHI, hiPCA, Shannon_entropy
    """
    metadata_cols = ['Diagnosis', 'Project', 'GMHI', 'hiPCA', 'Shannon_entropy']
    feature_columns = [col for col in merged_data.columns if col not in metadata_cols]
    logging.info(f"Identified {len(feature_columns)} feature columns.")
    return feature_columns
