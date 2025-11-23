# src/models/evaluate.py

import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.features.feature_engineering import create_basic_features

DATA_PATH = "E:\\shadowfox\\phase2\\car_pred\\data\\car.csv"

# model paths
MODEL1 = "model\best_model.joblib"
MODEL2 = "model\best_tuned_model.joblib"


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = create_basic_features(df)

    # remove columns not used in training
    df = df.drop(columns=['Car_Name', 'Year'], errors='ignore')

    X = df.drop(columns=['Selling_Price'])
    y = df['Selling_Price']
    return X, y


def evaluate_model(model_path, X, y):
    if not os.path.exists(model_path):
        print(f"[SKIP] Model not found: {model_path}")
        return None

    print(f"\nLoading model: {model_path}")
    model = joblib.load(model_path)

    preds = model.predict(X)

    rmse = mean_squared_error(y, preds, squared=False)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print(f"Results for {model_path}")
    print(f" RMSE : {rmse:.4f}")
    print(f" MAE  : {mae:.4f}")
    print(f" R²   : {r2:.4f}")

    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    print("Loading dataset...")
    X, y = load_dataset()

    print("\nEvaluating Models...\n")

    results = {}

    results["default_model"] = evaluate_model(MODEL1, X, y)
    results["tuned_model"] = evaluate_model(MODEL2, X, y)

    print("\n\n=== FINAL SUMMARY ===")
    for name, metrics in results.items():
        if metrics:
            print(f"\n{name.upper()}:")
            print(f" RMSE = {metrics['rmse']:.4f}")
            print(f" MAE  = {metrics['mae']:.4f}")
            print(f" R2   = {metrics['r2']:.4f}")
        else:
            print(f"\n{name.upper()}: Not available")


if __name__ == "__main__":
    main()
