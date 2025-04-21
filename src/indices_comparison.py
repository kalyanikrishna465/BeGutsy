import pandas as pd
import numpy as np
import logging
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def compare_indices(merged_data, computed_index, index_name="New_Index"):
    """
    Compare the computed health index with existing indices from metadata (GMHI, hiPCA, Shannon_entropy).
    Computes Pearson correlation coefficients.
    Returns a dictionary with correlations.
    """
    try:
        comparisons = {}
        existing_indices = ["GMHI", "hiPCA", "Shannon_entropy"]
        for col in existing_indices:
            if col in merged_data.columns:
                corr, pval = pearsonr(merged_data[col].values, computed_index)
                comparisons[col] = {"correlation": corr, "p_value": pval}
                logging.info(f"Correlation between {index_name} and {col}: {corr} (p={pval})")
        return comparisons
    except Exception as e:
        logging.error(f"Error in compare_indices: {e}")
        raise e

def plot_index_comparisons(merged_data, computed_index, output_file="comparison.png"):
    """
    Create scatter plots comparing the computed index with each of the existing indices.
    """
    try:
        existing_indices = ["GMHI", "hiPCA", "Shannon_entropy"]
        num_plots = len(existing_indices)
        fig, axs = plt.subplots(1, num_plots, figsize=(5*num_plots, 4))
        if num_plots == 1:
            axs = [axs]
        for ax, col in zip(axs, existing_indices):
            if col in merged_data.columns:
                ax.scatter(merged_data[col].values, computed_index, alpha=0.6)
                ax.set_xlabel(col)
                ax.set_ylabel("Computed Health Index")
                ax.set_title(f"{col} vs Computed Index")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        logging.info(f"Index comparison plot saved to {output_file}")
    except Exception as e:
        logging.error(f"Error in plot_index_comparisons: {e}")
        raise e
