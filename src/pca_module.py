import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import PCA

def compute_pca(features_df, variance_explained=0.90, n_components=None):
    """
    Perform PCA on the feature dataframe.
    If n_components is not provided, determine the number of components required
    to reach the desired variance explained.
    Returns:
      pca_model: trained PCA model
      transformed_data: PCA scores for the samples
    """
    try:
        pca_model = PCA(n_components=n_components)
        transformed_data = pca_model.fit_transform(features_df.values)
        
        if n_components is None:
            cum_var = np.cumsum(pca_model.explained_variance_ratio_)
            n_components_required = np.searchsorted(cum_var, variance_explained) + 1
            logging.info(f"Number of components required to explain {variance_explained*100}% variance: {n_components_required}")
            pca_model = PCA(n_components=n_components_required)
            transformed_data = pca_model.fit_transform(features_df.values)
        
        logging.info(f"PCA completed with {pca_model.n_components_} components.")
        return pca_model, transformed_data
    except Exception as e:
        logging.error(f"Error in compute_pca: {e}")
        raise e

def reconstruct_data(pca_model, transformed_data):
    """
    Reconstruct data using PCA model.
    Returns the reconstructed data (in original feature space).
    """
    try:
        reconstructed = pca_model.inverse_transform(transformed_data)
        logging.info("Data reconstruction from PCA completed.")
        return reconstructed
    except Exception as e:
        logging.error(f"Error in reconstruct_data: {e}")
        raise e

def compute_residuals(features_df, reconstructed):
    """
    Compute residuals as the difference between original and reconstructed data.
    """
    try:
        residuals = features_df.values - reconstructed
        logging.info("Computed residuals from PCA reconstruction.")
        return residuals
    except Exception as e:
        logging.error(f"Error in compute_residuals: {e}")
        raise e

def compute_T2_index(sample, pca_model):
    """
    Compute Hotelling's T^2 index for a given sample.
    T2 = t^T * Lambda^{-1} * t, where t is the PCA score vector for the sample.
    """
    try:
        t = pca_model.transform(sample.reshape(1, -1))[0]
        inv_lambda = np.diag(1.0 / np.where(pca_model.explained_variance_ > 0,
                                              pca_model.explained_variance_,
                                              1e-6))
        T2 = np.dot(np.dot(t.T, inv_lambda), t)
        return T2
    except Exception as e:
        logging.error(f"Error in compute_T2_index: {e}")
        raise e

def compute_Q_index(sample, pca_model):
    """
    Compute Q index (squared prediction error) for a given sample.
    Q = || x - x_hat ||^2, where x_hat is the reconstruction of x from PCA.
    """
    try:
        x_hat = pca_model.inverse_transform(pca_model.transform(sample.reshape(1, -1)))[0]
        Q = np.linalg.norm(sample - x_hat) ** 2
        return Q
    except Exception as e:
        logging.error(f"Error in compute_Q_index: {e}")
        raise e

def compute_control_limits(pca_model, features_df, confidence_level=0.95):
    """
    Compute control limits for T2 and Q charts.
    
    For T2: 
      T2_limit = chi2.ppf(confidence_level, df = n_components)
    
    For Q:
      1. Compute the full covariance matrix of the original data X.
      2. Compute its eigenvalues (sorted in descending order).
      3. Let d = number of retained components (pca_model.n_components_),
         and let the residual eigenvalues be those from index d to D.
      4. Compute:
            theta1 = sum_{i=d+1}^{D} lambda_i
            theta2 = sum_{i=d+1}^{D} lambda_i^2
            theta3 = sum_{i=d+1}^{D} lambda_i^3
         Then h0 = 1 - (2 * theta1 * theta3) / (3 * theta2**2)
         and Q_limit = theta1 * (((chi2.ppf(confidence_level, df=1))**(2*theta2/theta3) - 1)/h0 + 1)
         If h0 is less than or equal to zero, then set Q_limit = theta1.
    
    Returns:
      T2_limit, Q_limit
    """
    try:
        from scipy.stats import chi2
        
        d = pca_model.n_components_
        T2_limit = chi2.ppf(confidence_level, df=d)
        
        # Compute the full covariance matrix of the original features
        X = features_df.values
        cov_matrix = np.cov(X, rowvar=False)
        
        # Compute full set of eigenvalues of the covariance matrix (ensure real values)
        eigvals_full, _ = np.linalg.eigh(cov_matrix)
        # Sort eigenvalues in descending order
        eigvals_full = np.sort(eigvals_full)[::-1]
        
        total_features = len(eigvals_full)
        if d < total_features:
            residual_eigvals = eigvals_full[d:]
            theta1 = np.sum(residual_eigvals)
            theta2 = np.sum(residual_eigvals**2)
            theta3 = np.sum(residual_eigvals**3)
            if theta2 == 0:
                h0 = 1e-6
            else:
                h0 = 1 - (2 * theta1 * theta3) / (3 * (theta2**2))
            if h0 <= 0:
                logging.warning("h0 <= 0 encountered; setting Q_limit = theta1 as fallback.")
                Q_limit = theta1
            else:
                Q_limit = theta1 * (((chi2.ppf(confidence_level, df=1))**(2*theta2/theta3) - 1) / h0 + 1)
        else:
            Q_limit = 0.0
        
        logging.info(f"Control limits computed: T2_limit = {T2_limit}, Q_limit = {Q_limit}")
        return T2_limit, Q_limit
    except Exception as e:
        logging.error(f"Error in compute_control_limits: {e}")
        raise e

def compute_combined_index(T2, Q, T2_limit, Q_limit):
    """
    Compute the combined health index as a function of T2 and Q indexes.
    For example, one formulation:
      phi = sqrt((T2 / T2_limit)^2 + (Q / Q_limit)^2)
    """
    try:
        if T2_limit == 0 or Q_limit == 0:
            combined = np.nan
        else:
            combined = np.sqrt((T2 / T2_limit)**2 + (Q / Q_limit)**2)
        return combined
    except Exception as e:
        logging.error(f"Error in compute_combined_index: {e}")
        raise e
