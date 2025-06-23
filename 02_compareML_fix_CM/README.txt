# CDOM Prediction using Machine Learning Models

This project builds machine learning models to estimate Colored Dissolved Organic Matter (CDOM) concentration using spectral band ratios. It includes preprocessing, level-aware splitting, hyperparameter optimization, and performance evaluation.

## Models Included
- **SVR** (Support Vector Regression)
- **Random Forest**
- **XGBoost**
- **ANN (PyTorch)**

## Features
- Level-aware split (Level 1 & Level 2)
- MinMax normalization
- Optuna hyperparameter tuning
- Accuracy metrics: R², MAE, RMSE
- Auto-saving predictions, trained models, and plots

## How to Run

1. Prepare your dataset:
   - Excel file named `new_dataset.xlsx`
   - Sheet name must be `"r2"`
   - Target column: `a355`
   - Classification column: `Level`

2. Install dependencies:
```bash
pip install -r requirements.txt
