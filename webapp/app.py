import os
import sys
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Import project modules
from preprocessing import preprocess_features
from pca_module import compute_T2_index, compute_Q_index, compute_combined_index

# Flask app
app = Flask(__name__, template_folder='templates')

# Load PCA model and selected features
model_dir = os.path.join("..", "output", "20250406_171218")
pca_model = joblib.load(os.path.join(model_dir, "pca_model.pkl"))
selected_features = joblib.load(os.path.join(model_dir, "selected_features.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["csv_file"]
        if file:
            user_df = pd.read_csv(file)

            # Select only relevant features
            user_df = user_df[selected_features]

            # Preprocess and handle NaNs
            user_features = preprocess_features(user_df, selected_features)
            user_features = user_features.fillna(0)  # Fix: fill NaNs to avoid PCA errors

            # Compute indexes for the first sample
            sample = user_features.iloc[0].values
            
            # Placeholder values (replace this block with real logic later)
            T2 = 3.562
            Q = 0.823
            combined_index = 0.95
            status = "Healthy"

            result = {
                "T2": round(T2, 3),
                "Q": round(Q, 3),
                "Combined Index": round(combined_index, 3),
                "Predicted Status": status
            }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
