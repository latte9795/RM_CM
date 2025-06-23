This repository contains three experimental pipelines for comparing machine learning models in predicting water quality indicators such as CDOM and CM using hyperspectral or satellite-derived features.

📁 Repository Structure

Folder

Purpose

Notes

01_compareML

Base model comparison for CDOM

Default version for evaluating SVR, RFR, XGB, ANN on CDOM prediction

02_compareML_fix_CM

Model comparison for CM

Target variable changed to CM (e.g., Chlorophyll-a or a similar indicator), features are the same

03_compareML_fix_RM

RMSE-sorted visualization & evaluation

Maintains CM as target, focuses on RMSE-aligned result presentation

Each folder contains independent pipelines for training, optimizing, evaluating, and visualizing multiple regression models.

🚀 Getting Started

1. Requirements

Install dependencies using the following command:

pip install -r requirements.txt

2. Input Format

Excel file: new_dataset.xlsx

Sheet: r2

Target: a355 (CDOM) or CM (based on folder)

Must include Level column for stratified sampling

3. Folder Contents

├── cdom_model.py            # Main script
├── new_dataset.xlsx         # Input data
├── M01_SVR_model.csv        # SVR results
├── M02_RFR_model.csv        # RFR results
├── M03_XGB_model.csv        # XGB results
├── M04_ANN_model.csv        # ANN results
├── result/                  # CSV prediction outputs
├── graph/                   # Scatter plots
└── temp/                    # Saved models (.pkl / .pth)

🧠 Model Pipeline Overview

Each folder performs the following steps:

Load and preprocess dataset (MinMax normalization)

Level-aware train/test splitting

Model training with Optuna-based hyperparameter tuning

Evaluation: R², MAE, RMSE

Save predictions, plots, and trained models

Supported Models:

SVR: Support Vector Regression (scikit-learn)

RFR: Random Forest Regression

XGB: XGBoost Regression

ANN: Artificial Neural Network (PyTorch)

📊 Result Interpretation

Each CSV file (M0X_*.csv) contains train/test performance per iteration (R², MAE, RMSE) along with best hyperparameters

Scatter plots (graph/*.png) visualize predicted vs. observed values

Results can be compared across folders to observe:

Model stability over iterations

Effect of target variable change (CDOM vs. CM)

Alignment impact (RMSE sorting)

🔍 Example Use Cases

Evaluate suitability of ML algorithms for aquatic carbon proxy estimation

Benchmark predictive stability using fixed-level splits

Analyze preprocessing or visualization improvements by comparing across folders
