import pandas as pd
import numpy as np
import logging
from scipy.stats import ks_2samp

def select_features_ks(preprocessed_df, merged_data, diagnosis_col='Diagnosis', pvalue_threshold=0.001):
    """
    Select features using the Kolmogorov-Smirnov test.
    Compare distribution of each feature in healthy vs diseased samples.
    Returns list of selected feature names.
    """
    try:
        healthy = merged_data[merged_data[diagnosis_col].str.lower() == "healthy"]
        diseased = merged_data[merged_data[diagnosis_col].str.lower() != "healthy"]
        
        selected_features = []
        for feature in preprocessed_df.columns:
            healthy_vals = preprocessed_df.loc[healthy.index, feature].dropna()
            diseased_vals = preprocessed_df.loc[diseased.index, feature].dropna()
            if len(healthy_vals) < 10 or len(diseased_vals) < 10:
                continue
            stat, pvalue = ks_2samp(healthy_vals, diseased_vals)
            if pvalue < pvalue_threshold:
                selected_features.append(feature)
        logging.info(f"Selected {len(selected_features)} features using KS test with p-value threshold {pvalue_threshold}.")
        return selected_features
    except Exception as e:
        logging.error(f"Error in select_features_ks: {e}")
        raise e
