import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.features.feature_engineering import create_basic_features
from src.features.preprocessing import build_preprocessor

DATA_PATH = "data\car.csv"
OUT_PATH = "model\best_tuned_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_and_prep():
    df = pd.read_csv(DATA_PATH)
    df = create_basic_features(df)
    df = df.drop(columns=['Car_Name','Year'], errors='ignore')
    X = df.drop(columns=['Selling_Price'])
    y = df['Selling_Price']
    return X, y

def main():
    X, y = load_and_prep()

    numeric_features = ['Present_Price', 'Kms_Driven', 'Age',
                        'KM_per_Year', 'Price_Depreciation', 'Car_Condition']
    categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission',
                            'Brand', 'Is_First_Owner', 'Is_Diesel']
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', rf)
    ])

    param_dist = {
        'model__n_estimators': [200, 400, 600, 800],
        'model__max_depth': [None, 6, 10, 20, 30],
        'model__min_samples_split': [2, 4, 6, 8],
        'model__min_samples_leaf': [1, 2, 4, 6],
        'model__max_features': ['sqrt', 'log2', 0.5, 0.8]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=30,
        scoring='neg_root_mean_squared_error',
        cv=5,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("Starting RandomizedSearchCV...")
    search.fit(X, y)
    print("Best params:", search.best_params_)
    print("Best CV RMSE:", -search.best_score_)

    best_pipeline = search.best_estimator_

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump(best_pipeline, OUT_PATH)

    print("Saved best tuned pipeline to:", OUT_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    preds = best_pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Holdout RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")

if __name__ == '__main__':
    main()
