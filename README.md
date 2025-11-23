# shadowfox - Car Price Prediction (Intermediate Task)

This repository contains a complete pipeline for second-hand car price prediction.

**Dataset**: place your CSV at `/mnt/data/car.csv` (already present in this environment).

## Structure
- `src/features/feature_engineering.py` - feature engineering utilities  
- `src/features/preprocessing.py` - preprocessor builder  
- `src/models/train.py` - train and save a RandomForest pipeline  
- `src/models/train_tune.py` - hyperparameter tuning with RandomizedSearchCV  
- `model/` - saved models (created after running scripts)  
- `notebooks/EDA.ipynb` - place for exploratory analysis  

## How to run

1. Train default model:
```bash
python3 src/models/train.py
