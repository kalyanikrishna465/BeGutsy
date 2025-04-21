import numpy as np
import pandas as pd
import logging

def compute_M_matrix(pca_model, T2_limit, Q_limit, n_features):
    """
    Compute the matrix M used in the combined index derivation.
    For demonstration, we define M as a weighted sum of the projection matrices:
    M = (T2_limit)*P_latent*P_latent^T + (Q_limit)*P_residual*P_residual^T.
    Here, P_latent are the PCA components (n_components x n_features) and
    P_residual is approximated as the orthogonal complement.
    For simplicity, we compute an approximate M.
    """
    try:
        P_latent = pca_model.components_
        n_components = pca_model.n_components_
        Proj_latent = np.dot(P_latent.T, P_latent)
        I = np.eye(n_features)
        Proj_residual = I - Proj_latent
        M = T2_limit * Proj_latent + Q_limit * Proj_residual
        return M
    except Exception as e:
        logging.error(f"Error in compute_M_matrix: {e}")
        raise e

def compute_bhc_contributions(sample, M):
    """
    Compute the bacteria-to-health contribution (BHC) for each feature in a given sample.
    Following the derivation:
      For each feature i, let e_i be the unit vector in the i-th direction.
      Then contribution f_i = (e_i^T M x) / (e_i^T M e_i)
    Returns a dictionary mapping feature index to its contribution.
    """
    try:
        contributions = {}
        x = sample
        for i in range(len(x)):
            e_i = np.zeros(len(x))
            e_i[i] = 1
            numerator = np.dot(e_i, np.dot(M, x))
            denominator = np.dot(e_i, np.dot(M, e_i))
            if denominator == 0:
                contributions[i] = 0.0
            else:
                contributions[i] = numerator / denominator
        return contributions
    except Exception as e:
        logging.error(f"Error in compute_bhc_contributions: {e}")
        raise e

def analyze_population_bhc(features_df, pca_model, T2_limit, Q_limit):
    """
    For each sample in the feature dataframe, compute the BHC contributions.
    Returns a DataFrame where rows are samples and columns are features with their contributions.
    """
    try:
        n_features = features_df.shape[1]
        M = compute_M_matrix(pca_model, T2_limit, Q_limit, n_features)
        bhc_results = {}
        for idx, row in features_df.iterrows():
            contributions = compute_bhc_contributions(row.values, M)
            bhc_results[idx] = contributions
        bhc_df = pd.DataFrame.from_dict(bhc_results, orient='index')
        logging.info("BHC analysis completed for all samples.")
        return bhc_df
    except Exception as e:
        logging.error(f"Error in analyze_population_bhc: {e}")
        raise e
