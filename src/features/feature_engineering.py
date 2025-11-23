import pandas as pd
import numpy as np

CURRENT_YEAR = 2025

def extract_brand(df, car_name_col='Car_Name'):
    if car_name_col in df.columns:
        return df[car_name_col].astype(str).apply(lambda x: x.split()[0])
    else:
        return pd.Series(['Unknown'] * len(df), index=df.index)

def create_basic_features(df):
    """Create mandatory and instructor-requested engineered features."""
    df = df.copy()

    # Age
    if 'Year' in df.columns:
        df['Age'] = CURRENT_YEAR - df['Year']
    else:
        df['Age'] = np.nan

    # Brand
    df['Brand'] = extract_brand(df)

    # KM per year
    df['KM_per_Year'] = df['Kms_Driven'] / (df['Age'].replace(0, np.nan) + 1)
    df['KM_per_Year'] = df['KM_per_Year'].fillna(df['Kms_Driven'])

    # Price depreciation (simple proxy)
    df['Price_Depreciation'] = df['Present_Price'] / (df['Age'] + 1)

    # Car condition score (heuristic)
    df['Car_Condition'] = (df['Present_Price'] / (df['Kms_Driven'] + 1)) * (1.0 / (df['Age'] + 1))

    # Binary flags
    df['Is_Diesel'] = (df['Fuel_Type'].str.lower() == 'diesel').astype(int)
    df['Is_First_Owner'] = (df['Owner'] == 0).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df
