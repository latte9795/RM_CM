# CDOM Prediction using Machine Learning

This project applies various machine learning models (SVR, Random Forest, XGBoost, ANN) to estimate CDOM (Colored Dissolved Organic Matter) values using optical satellite bands.

## Features

- Train/Test split by Level 1 and Level 2
- Optuna-based hyperparameter optimization
- Support for SVR, RFR, XGB, ANN (PyTorch)
- Saves `.pkl`, `.pth` models and prediction plots
- Evaluation metrics: R², MAE, RMSE

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
