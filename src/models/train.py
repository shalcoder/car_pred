import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.features.feature_engineering import create_basic_features
from src.features.preprocessing import build_preprocessor

DATA_PATH = "data\car.csv"
MODEL_OUT = "model\best_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def prepare_data(df):
    df = create_basic_features(df)

    drop_cols = []
    if 'Car_Name' in df.columns:
        drop_cols.append('Car_Name')
    if 'Year' in df.columns:
        drop_cols.append('Year')

    df = df.drop(columns=drop_cols, errors='ignore')

    y = df['Selling_Price']
    X = df.drop(columns=['Selling_Price'])

    return X, y

def main():
    df = load_data()
    X, y = prepare_data(df)

    numeric_features = ['Present_Price', 'Kms_Driven', 'Age',
                        'KM_per_Year', 'Price_Depreciation', 'Car_Condition']
    categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission',
                            'Brand', 'Is_First_Owner', 'Is_Diesel']

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    print("Evaluating...")
    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT)

    print("Saved trained pipeline to:", MODEL_OUT)

if __name__ == '__main__':
    main()
