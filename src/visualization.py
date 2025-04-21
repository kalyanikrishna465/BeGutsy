import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import logging

def plot_pca_scores(transformed_data, labels, output_file="pca_scores.png"):
    """
    Generate a static scatter plot of the first two PCA scores.
    """
    try:
        plt.figure(figsize=(8,6))
        plt.scatter(transformed_data[:,0], transformed_data[:,1], c=labels, cmap="viridis", alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Scores")
        plt.colorbar(label="Label")
        plt.savefig(output_file)
        plt.close()
        logging.info(f"PCA scores plot saved to {output_file}")
    except Exception as e:
        logging.error(f"Error in plot_pca_scores: {e}")
        raise e

def plot_control_chart(sample_control_values, sample_id, output_file="control_chart.png"):
    """
    Generate a static plot showing T2 and Q index for a given sample.
    """
    try:
        labels = list(sample_control_values.keys())
        values = list(sample_control_values.values())
        plt.figure(figsize=(6,4))
        plt.bar(labels, values)
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title(f"Control Chart Indices for Sample {sample_id}")
        plt.savefig(output_file)
        plt.close()
        logging.info(f"Control chart plot for sample {sample_id} saved to {output_file}")
    except Exception as e:
        logging.error(f"Error in plot_control_chart: {e}")
        raise e

def interactive_pca_plot(transformed_data, labels):
    """
    Generate an interactive scatter plot of PCA scores using Plotly.
    Returns the Plotly figure.
    """
    try:
        df = pd.DataFrame({
            "PC1": transformed_data[:, 0],
            "PC2": transformed_data[:, 1],
            "Label": labels
        })
        fig = px.scatter(df, x="PC1", y="PC2", color="Label", title="Interactive PCA Scores")
        return fig
    except Exception as e:
        logging.error(f"Error in interactive_pca_plot: {e}")
        raise e
