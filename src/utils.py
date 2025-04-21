import json
import logging
import os
import sys
import argparse
import joblib

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        sys.exit(1)

def setup_logging(log_file, log_level="DEBUG"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))
    
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Gut Microbiome-based Health Index Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file")
    args = parser.parse_args()
    return args

def safe_run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {e}")
        raise e
    
def save_model(model, filepath):
    """
    Save a scikit-learn model to disk using joblib.
    """
    try:
        joblib.dump(model, filepath)
        logging.info(f"Model saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving model to {filepath}: {e}")
        raise e

def load_model(filepath):
    """
    Load a scikit-learn model from disk using joblib.
    """
    try:
        model = joblib.load(filepath)
        logging.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {filepath}: {e}")
        raise e
