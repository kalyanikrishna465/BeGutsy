import os
import datetime
import numpy as np
import pandas as pd
import logging
from utils import load_config, setup_logging, parse_arguments, safe_run
from data_loader import load_metadata, load_taxonomy, load_pathways, merge_data
from preprocessing import get_feature_columns, preprocess_features
from feature_selection import select_features_ks
from pca_module import compute_pca, reconstruct_data, compute_residuals, compute_T2_index, compute_Q_index, compute_control_limits, compute_combined_index
from control_chart import evaluate_sample_control_charts
from classification import classify_samples, evaluate_classification
from bhc_analysis import analyze_population_bhc
from indices_comparison import compare_indices, plot_index_comparisons
from visualization import plot_pca_scores, interactive_pca_plot
import matplotlib.pyplot as plt

def main():
    # Parse command-line arguments and load configuration
    args = parse_arguments()
    config = load_config(args.config)
    
    # Create a timestamped output directory and update log file path accordingly
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join("logs", f"{timestamp}.log")
    config["logging"]["log_file"] = log_file
    setup_logging(log_file, config["logging"]["log_level"])
    logging.info("Starting Gut Microbiome-based Health Index Pipeline.")
    
    # Load data
    metadata = safe_run(load_metadata, config["data"]["metadata_file"])
    taxonomy = safe_run(load_taxonomy, config["data"]["taxonomy_file"])
    pathways = safe_run(load_pathways, config["data"]["pathways_file"])
    
    merged_data = safe_run(merge_data, taxonomy, pathways, metadata)
    
    # Preprocess features
    feature_columns = get_feature_columns(merged_data)
    features_df = preprocess_features(merged_data, feature_columns)
    
    # Feature selection (using KS test)
    selected_features = select_features_ks(features_df, merged_data, pvalue_threshold=config["feature_selection"]["ks_pvalue_threshold"])
    if not selected_features:
        logging.error("No features selected. Exiting.")
        return
    features_selected = features_df[selected_features]
    logging.info(f"Feature matrix shape after selection: {features_selected.shape}")
    
    # PCA computation
    pca_model, pca_scores = compute_pca(features_selected, variance_explained=config["pca"]["variance_explained"],
                                        n_components=config["pca"]["n_components"])
    
    # Save the PCA model and selected features for later use
    model_filepath = os.path.join(output_dir, "pca_model.pkl")
    features_filepath = os.path.join(output_dir, "selected_features.pkl")
    from utils import save_model
    save_model(pca_model, model_filepath)
    save_model(selected_features, features_filepath)  # joblib can save Python lists too
    
    # Compute control limits using the full data (features_selected) for residual eigenvalue computation
    T2_limit, Q_limit = compute_control_limits(pca_model, features_selected, confidence_level=config["control_chart"]["confidence_level"])
    
    # Evaluate control charts for each sample and compute combined index
    combined_indices = []
    for i in range(features_selected.shape[0]):
        sample = features_selected.iloc[i].values
        control_vals = evaluate_sample_control_charts(sample, pca_model, T2_limit, Q_limit)
        combined_indices.append(control_vals["combined_index"])
      
    merged_data['Computed_Index'] = combined_indices
    
    # Classification
    threshold = config["classification"]["combined_index_threshold"]
    predictions = classify_samples(np.array(combined_indices), threshold)
    merged_data['Predicted_Label'] = predictions
    true_labels = merged_data['Diagnosis'].apply(lambda x: 1 if x.lower() == "healthy" else 0).values
    eval_metrics = evaluate_classification(true_labels, predictions)
    
    # BHC Analysis
    bhc_df = analyze_population_bhc(features_selected, pca_model, T2_limit, Q_limit)
    
    # Compare indices with existing ones
    comparisons = compare_indices(merged_data, np.array(combined_indices), index_name="Computed_Index")
    plot_index_comparisons(merged_data, np.array(combined_indices), output_file=os.path.join(output_dir, "comparison.png"))
    
    # Visualizations
    plot_pca_scores(pca_scores, true_labels, output_file=os.path.join(output_dir, "pca_scores.png"))
    if config["visualization"]["interactive"]:
        fig = interactive_pca_plot(pca_scores, true_labels)
        fig.write_html(os.path.join(output_dir, "interactive_pca_scores.html"))
        logging.info("Interactive PCA plot saved as interactive_pca_scores.html")
    
    # Save outputs
    merged_data.to_csv(os.path.join(output_dir, "merged_results.csv"), index=True)
    logging.info(f"Merged results saved to {os.path.join(output_dir, 'merged_results.csv')}")
    bhc_df.to_csv(os.path.join(output_dir, "bhc_results.csv"), index=True)
    logging.info(f"BHC results saved to {os.path.join(output_dir, 'bhc_results.csv')}")
    
    logging.info("Pipeline execution completed.")

if __name__ == "__main__":
    main()