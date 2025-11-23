import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import os

MODEL_PATH = r"model\best_tuned_model.joblib"
DATA_PATH = r"data\car.csv"

from src.features.feature_engineering import create_basic_features

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = create_basic_features(df)

    # Drop columns not used during training
    df = df.drop(columns=['Car_Name', 'Year'], errors="ignore")

    y = df['Selling_Price']
    X = df.drop(columns=['Selling_Price'])

    return X, y, df.columns

def plot_feature_importance(model, feature_names):
    importances = model["model"].feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,6))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)), np.array(feature_names)[idx], rotation=90)
    plt.title("RandomForest Feature Importance")
    plt.tight_layout()
    plt.show()

def shap_analysis(model, X):
    explainer = shap.TreeExplainer(model["model"])
    shap_values = explainer.shap_values(X)

    print("\nGenerating SHAP summary plot...")
    shap.summary_plot(shap_values, X)

def main():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded.")

    print("Loading data...")
    X, y, cols = load_data()
    print("Data loaded.")

    print("\n=== FEATURE IMPORTANCE (Model Built-In) ===")
    plot_feature_importance(model, X.columns)

    print("\n=== SHAP VALUES ===")
    shap_analysis(model, X.sample(200))  # sample to speed up

if __name__ == "__main__":
    main()
