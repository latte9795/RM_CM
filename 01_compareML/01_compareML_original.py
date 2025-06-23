# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
CDOM Prediction using SVR, RFR, XGBoost, and ANN with Optuna Optimization.

This script trains multiple machine learning models to estimate CDOM (Colored Dissolved Organic Matter)
concentrations using spectral band ratios. It performs level-aware train/val/test splitting,
normalization, model training, hyperparameter tuning, prediction, evaluation, and result export.

Author: latte9795/RM_CM/01_compareML
Created: 2025-06-23
"""

import os
import math
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from sklearn.preprocessing import MinMaxScaler
from numpy.polynomial.polynomial import polyfit
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches

def filter_nan(s,o):
    """
    this functions removed the data from simulated and observed data
    whereever the observed data contains nan
    this is used by all other functions, otherwise they will produce nan as output
    """
    #data = np.array([s.flatten(),o.flatten()])
    #data = np.transpose(data)
    #data = data[~np.isnan(data).any(1)]
    #return data[:,0],data[:,1]  # taking all rows (:) but keeping the second column (0, 1)
    x = []
    y = []
    for length in range(0, len(s)):
        if float(o[length]) != 0 and np.isnan(o[length]) == False:
            x.append(s[length])
            y.append(o[length])
    #print x[1:10], y[1:10]
    return x,y


def MinMaxScaler2(array):
    maxmin = []
    Max = 15  # PC 150, Chl-a 120
    Min = 0
    array = (array - Min) / (Max - Min)
    maxmin.append(Max)
    maxmin.append(Min)
    return maxmin, array


def NSE(SimulatedStreamFlow,ObservedStreamFlow):
    '''(SimulatedStreamFlow, ObservedStreamFlow)'''
    x = SimulatedStreamFlow
    y = ObservedStreamFlow
    A=0.0 #dominator
    B=0.0 #deminator
    tot = 0.0
    x,y = filter_nan(x,y)
    try:
        for i in range(0, len(y)):
            tot = tot + y[i]
        average = tot / len(y)
        for i in range(0, len(y)):
            A = A + math.pow((y[i] - x[i]), 2)
            B = B + math.pow((y[i] - average), 2)
        E = 1 - (A/B) # Nash-Sutcliffe model eficiency coefficient
    except:
        E = 0
    return E


def R(sim, obs):
    if type(sim) == np.ndarray and type(obs) == np.ndarray:
        sim = sim.ravel()
        obs = obs.ravel()
    x = sim
    y = obs
    x, y = filter_nan(x, y)
    if len(y) == 0:
        R = np.NaN
    else:
        # R = np.ma.corrcoef(np.array(obs).astype(np.float), np.array(sim).astype(np.float))[0, 1]
        R = np.ma.corrcoef(np.array(y).astype(float), np.array(x).astype(float))[0, 1]
    R2 = R ** 2
    return np.round(R2, 3)


def scatter_plot(title, obs_tr, sim_tr, obs_te, sim_te, obs_name, sim_name, out_plot):
    obs_tr, sim_tr = np.array(obs_tr).astype("float"), np.array(sim_tr).astype("float")
    intercept1, slope1 = polyfit(obs_tr, sim_tr, 1)
    poly1d1 = np.poly1d([slope1, intercept1])
    pearson1 = R(obs_tr, sim_tr)
    obs_te, sim_te = np.array(obs_te).astype("float"), np.array(sim_te).astype("float")
    intercept2, slope2 = polyfit(obs_te, sim_te, 1)
    poly1d2 = np.poly1d([slope2, intercept2])
    pearson2 = R(obs_te, sim_te)
    x = np.arange(0, 15)
    y = x
    # rmse = objective_function.RMSE(obs, sim)
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title, size=20)
    ax.set_xlabel(obs_name, size=16)
    ax.set_ylabel(sim_name, size=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.scatter(obs_tr, sim_tr, color='blue', edgecolors='black')
    ax.scatter(obs_te, sim_te, color='orange', edgecolors='black')
    # ax.plot(obs, poly1d(obs), color="red", linestyle="solid", linewidth=0.5, label="fit")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    legend_handle1 = mpatches.Patch(color='blue')  # 표식의 색깔 설정
    legend_handle2 = mpatches.Patch(color='orange')
    ax.legend(handles=[legend_handle1, legend_handle2], labels=['train', 'test'], loc=2)
    poly_1 = str(poly1d1).replace('\n', '')
    poly_2 = str(poly1d2).replace('\n', '')
    ax.add_artist(AnchoredText(f"Train $R^2$: {round(pearson1, 3)}\ny={poly_1}\n"
                               f"Test $R^2$: {round(pearson2, 3)}\ny={poly_2}", loc=1, prop=dict(size=10)))

    plt.plot(obs_tr, slope1 * obs_tr + intercept1, color="blue", linestyle="--", alpha=0.3)
    plt.plot(obs_te, slope2 * obs_te + intercept2, color="orange", linestyle="--", alpha=0.3)
    plt.plot(x, y, color='black', linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close('all')


def svr_modeling(xx_train, x_val, x_train, x_test, yy_train, y_val, y_train, method, iter):
    def objective_svr(trial):
        C = trial.suggest_float('C', 1, 100, log=True)
        gamma = trial.suggest_float('gamma', 1, 100, log=True)
        epsilon = trial.suggest_float('epsilon', 0.001, 0.01)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid'])
        # 모델 생성 및 학습
        model = SVR(C=C,
                    gamma=gamma,
                    epsilon=epsilon,
                    kernel=kernel
                    )
        model.fit(xx_train, yy_train.ravel())

        # 검증
        y_pred_tr = model.predict(xx_train)
        y_pred_vl = model.predict(x_val)
        tr_rmse = math.sqrt(mean_squared_error(yy_train, y_pred_tr))
        vl_rmse = math.sqrt(mean_squared_error(y_val, y_pred_vl))
        rmse = (tr_rmse + vl_rmse) / 2
        return rmse

    study_svr = optuna.create_study(direction='minimize')
    study_svr.optimize(objective_svr, n_trials=100)
    best_hyperparams_svr = study_svr.best_trial.params
    svr_model_best = SVR(**best_hyperparams_svr)
    svr_model_best.fit(x_train, y_train.ravel())
    # 최적화된 모델 저장
    model_path = os.path.join(os.getcwd(), "temp", "02_SVR", f"SVR_CDOM_{method}_{iter}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(svr_model_best, f)

    # 학습 및 테스트 데이터에 대한 예측 수행
    y_SVR_tr = svr_model_best.predict(x_train)
    y_SVR_te = svr_model_best.predict(x_test)

    C = best_hyperparams_svr['C']
    gamma = best_hyperparams_svr['gamma']
    epsilon = best_hyperparams_svr['epsilon']
    kernel = best_hyperparams_svr['kernel']

    return y_SVR_tr, y_SVR_te, C, gamma, epsilon, kernel


def rfr_modeling(xx_train, x_val, x_train, x_test, yy_train, y_val, y_train, method, iter):
    def objective_rfr(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 10, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 4, 6)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 4, 8)
        max_features = trial.suggest_float('max_features', 0.7, 0.85)

        # 모델 생성 및 학습
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features
        )

        model.fit(xx_train, yy_train.ravel())  # 모델 학습

        # 검증
        y_pred_tr = model.predict(xx_train)
        y_pred_vl = model.predict(x_val)
        tr_rmse = math.sqrt(mean_squared_error(yy_train, y_pred_tr))
        vl_rmse = math.sqrt(mean_squared_error(y_val, y_pred_vl))
        rmse = (tr_rmse + vl_rmse) / 2
        return rmse

    study_rfr = optuna.create_study(direction='minimize')
    study_rfr.optimize(objective_rfr, n_trials=100)
    best_hyperparams_rfr = study_rfr.best_trial.params
    rfr_model_best = RandomForestRegressor(**best_hyperparams_rfr)
    rfr_model_best.fit(x_train, y_train.ravel())
    # 최적화된 모델 저장
    model_path = os.path.join(os.getcwd(), "temp", "03_RFR", f"RFR_CDOM_{method}_{iter}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(rfr_model_best, f)

    # 학습 및 테스트 데이터에 대한 예측 수행
    y_RFR_tr = rfr_model_best.predict(x_train)
    y_RFR_te = rfr_model_best.predict(x_test)

    NE_R = best_hyperparams_rfr['n_estimators']
    MD_R = best_hyperparams_rfr['max_depth']
    MSS_R = best_hyperparams_rfr['min_samples_split']
    MSL_R = best_hyperparams_rfr['min_samples_leaf']
    MF_R = best_hyperparams_rfr['max_features']

    return y_RFR_tr, y_RFR_te, NE_R, MD_R, MSS_R, MSL_R, MF_R


def xgb_modeling(xx_train, x_val, x_train, x_test, yy_train, y_val, y_train, method, iter):
    def objective_xgb(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 5, 8)
        subsample = trial.suggest_float('subsample', 0.8, 0.9)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.7, 0.9)
        min_child_weight = trial.suggest_int('min_child_weight', 5, 10)
        learning_rate = trial.suggest_float('learning_rate', 1e-2, 0.1, log=True)  # = eta와 같은값

        # 모델 생성 및 학습
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            learning_rate=learning_rate
        )
        model.fit(xx_train, yy_train.ravel(), early_stopping_rounds=20,
                  eval_set=[(x_val, y_val)], verbose=False)  # 모델 학습

        # 검증
        y_pred_tr = model.predict(xx_train)
        y_pred_vl = model.predict(x_val)
        tr_rmse = math.sqrt(mean_squared_error(yy_train, y_pred_tr))
        vl_rmse = math.sqrt(mean_squared_error(y_val, y_pred_vl))

        rmse = (tr_rmse + vl_rmse) / 2
        return rmse

    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(objective_xgb, n_trials=100)
    best_hyperparams_xgb = study_xgb.best_trial.params
    xgb_model_best = XGBRegressor(**best_hyperparams_xgb)
    xgb_model_best.fit(x_train, y_train.ravel())
    # 최적화된 모델 저장
    model_path = os.path.join(os.getcwd(), "temp", "04_XGB", f"XGB_CDOM_{method}_{iter}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(xgb_model_best, f)

    # 학습 및 테스트 데이터에 대한 예측 수행
    y_XGB_tr = xgb_model_best.predict(x_train)
    y_XGB_te = xgb_model_best.predict(x_test)

    NE_X = best_hyperparams_xgb['n_estimators']
    MD_X = best_hyperparams_xgb['max_depth']
    SS_X = best_hyperparams_xgb['subsample']
    CB_X = best_hyperparams_xgb['colsample_bytree']
    MC_X = best_hyperparams_xgb['min_child_weight']
    LR_X = best_hyperparams_xgb['learning_rate']

    return y_XGB_tr, y_XGB_te, NE_X, MD_X, SS_X, CB_X, MC_X, LR_X

# ANN 모델 클래스 정의
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_activation, output_activation):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # 은닉층 활성화 함수 설정
        self.hidden_activation = self.get_activation(hidden_activation)
        # 출력층 활성화 함수 설정
        self.output_activation = self.get_activation(output_activation)

    def forward(self, x):
        x = self.hidden_activation(self.fc1(x))
        x = self.output_activation(self.fc2(x))
        return x

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'ELU':
            return nn.ELU()
        elif activation == 'PReLU':
            return nn.PReLU()
        elif activation == 'SELU':
            return nn.SELU()
        elif activation == 'GELU':
            return nn.GELU()
        else:
            return nn.Identity()

# ANN 모델링 함수
# def ANN_modeling(x_train, y_train, x_test, trial, i, method):
def objective(trial, xx_train, x_val, yy_train, y_val, input_size):
    # 하이퍼파라미터 제안
    hidden_size = trial.suggest_int('hidden_size', 10, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
    hidden_activation = trial.suggest_categorical('hidden_activation', ['relu', 'leaky_relu', 'PReLU'])
    output_activation = trial.suggest_categorical('output_activation', ['leaky_relu', 'ELU', 'PReLU',
                                                                        'SELU', 'GELU', 'linear'])

    # 모델 생성
    model = ANN(input_size, hidden_size, 1, hidden_activation, output_activation)
    criterion = nn.MSELoss()   #MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 루프
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(torch.from_numpy(xx_train).float())
        loss = criterion(outputs, torch.from_numpy(yy_train.reshape(-1, 1)).float())
        loss.backward()
        optimizer.step()

    # 테스트 데이터 예측 및 평가
    model.eval()
    with torch.no_grad():
        y_tr_pred = model(torch.from_numpy(xx_train).float()).numpy()
        y_val_pred = model(torch.from_numpy(x_val).float()).numpy()
        tr_r2score = math.sqrt(mean_squared_error(yy_train, y_tr_pred))
        vl_r2score = math.sqrt(mean_squared_error(y_val, y_val_pred))
        rmse = (tr_r2score + vl_r2score) / 2
    return rmse


def accuracy_score(yTrain_re, yTest_re, y_Pred_tr_re, y_Pred_te_re):
    y_Pred_tr_re = y_Pred_tr_re.flatten()
    y_Pred_te_re = y_Pred_te_re.flatten()
    R2C = np.corrcoef(yTrain_re, y_Pred_tr_re)[0, 1] ** 2
    R2T = np.corrcoef(yTest_re, y_Pred_te_re)[0, 1] ** 2
    MAEC = mean_absolute_error(yTrain_re, y_Pred_tr_re)
    MAEV = mean_absolute_error(yTest_re, y_Pred_te_re)
    RMSEC = math.sqrt(mean_squared_error(yTrain_re, y_Pred_tr_re))
    RMSET = math.sqrt(mean_squared_error(yTest_re, y_Pred_te_re))
    return R2C, R2T, MAEC, MAEV, RMSEC, RMSET

# data loading
df = pd.read_excel(os.path.join('new_dataset.xlsx'), sheet_name='r2', header=0)

result_SVR = open(os.path.join(os.getcwd(), "M01_SVR_model.csv"), "w")
result_SVR.write("iteration,ML,method,train_r2,test_r2,train_MAE,test_MAE,train_RMSE,test_RMSE,"
                 "C,gamma,epsilon,kernel\n")
result_SVR.close()
result_RFR = open(os.path.join(os.getcwd(), "M02_RFR_model.csv"), "w")
result_RFR.write("iteration,ML,method,train_r2,test_r2,train_MAE,test_MAE,train_RMSE,test_RMSE,"
                 "n_estimators,max_depth,min_samples_split,min_samples_leaf,max_features\n")
result_RFR.close()
result_XGB = open(os.path.join(os.getcwd(), "M03_XGB_model.csv"), "w")
result_XGB.write("iteration,ML,method,train_r2,test_r2,train_MAE,test_MAE,train_RMSE,test_RMSE,"
                 "n_estimators,max_depth,subsample,colsample_bytree,min_child_weight,learning_rate\n")
result_XGB.close()
result_ANN = open(os.path.join(os.getcwd(), "M04_ANN_model.csv"), "w")
result_ANN.write("iteration,ML,method,train_r2,test_r2,train_MAE,test_MAE,train_RMSE,test_RMSE,"
                 "hidden_size,learning_rate,num_epochs,batch_size,hidden_activation_function,output_activation_function\n")
result_ANN.close()

# Input data normalization
Input = df.drop(['a355', 'Level'], axis=1)
Output_pre = df.drop(['Level', 'B05/B02', 'B05/B01', 'B05/B04', 'B05/B03', 'B03/B02', 'B08/B07'], axis=1)
ymaxmin, Output = MinMaxScaler2(Output_pre)
Output = pd.DataFrame(Output)

# Data merge
df_new = pd.concat([Output, df['Level'], Input], axis=1)

# Data separation for each level
level1 = (df_new.Level == 1)
level2 = (df_new.Level == 2)

df_L1 = df_new[level1]
df_L2 = df_new[level2]

Inx_L1 = math.floor(len(df_L1) * 0.2)
Inx_L2 = math.floor(len(df_L2) * 0.2)

iter = 0

while iter < 200:
    
    # Random sampling for each level
    df_rnd_L1 = df_L1.sample(frac=1).reset_index(drop=True)
    df_rnd_L2 = df_L2.sample(frac=1).reset_index(drop=True)
    
    # Separation between training and test dataset for each level
    df_te_L1 = df_rnd_L1.loc[:Inx_L1-1, :]
    df_te_L2 = df_rnd_L2.loc[:Inx_L2-1, :]
    df_vl_L1 = df_rnd_L1.loc[Inx_L1: 2 * Inx_L1 - 1, :]
    df_vl_L2 = df_rnd_L2.loc[Inx_L2: 2 * Inx_L2 - 1, :]
    df_tr_L1 = df_rnd_L1.loc[2 * Inx_L1:, :]
    df_tr_L2 = df_rnd_L2.loc[2 * Inx_L2:, :]

    df_tr_nor = pd.concat([df_tr_L1, df_tr_L2], axis=0)
    df_vl_nor = pd.concat([df_vl_L1, df_vl_L2], axis=0)
    df_te_nor = pd.concat([df_te_L1, df_te_L2], axis=0)

    df_train_x_ori = df_tr_nor.drop(['a355', 'Level'], axis=1)
    df_train_x_ori = df_train_x_ori.values
    df_train_y_ori = df_tr_nor['a355']
    df_train_y_ori = df_train_y_ori.values
    df_train_y_ori = np.array(df_train_y_ori).reshape(len(df_train_y_ori))

    # Over sampling method
    # Separation with L1, L2, and L3
    df_train_x_Level1 = (df_tr_nor.Level == 1)
    df_train_x_Level1 = df_tr_nor[df_train_x_Level1]
    df_train_x_L1 = df_train_x_Level1.drop(['Level'], axis=1)
    df_train_label_L1 = df_train_x_Level1[['Level']]
    df_train_x_Level2 = (df_tr_nor.Level == 2)
    df_train_x_Level2 = df_tr_nor[df_train_x_Level2]
    df_train_x_L2 = df_train_x_Level2.drop(['Level'], axis=1)
    df_train_label_L2 = df_train_x_Level2[['Level']]

    df_train_x_L1_L2 = pd.concat([df_train_x_L1, df_train_x_L2], axis=0)
    df_train_label_L1_L2 = pd.concat([df_train_label_L1, df_train_label_L2], axis=0)

    # Setup input and output for validation/test data
    df_vl_x_ori = df_vl_nor.drop(['a355', 'Level'], axis=1)
    x_val = df_vl_x_ori.values
    df_vl_y_ori = df_vl_nor[['a355']]
    df_vl_y_ori = df_vl_y_ori.values
    y_val = np.array(df_vl_y_ori).reshape(len(df_vl_y_ori))

    df_te_x_ori = df_te_nor.drop(['a355', 'Level'], axis=1)
    x_test = df_te_x_ori.values
    df_te_y_ori = df_te_nor[['a355']]
    df_te_y_ori = df_te_y_ori.values
    y_test = np.array(df_te_y_ori).reshape(len(df_te_y_ori))

    # data resampling
    # original training dataset
    df_Original_xtrain = np.vstack((df_train_x_ori, x_val))
    df_Original_ytrain = np.concatenate((df_train_y_ori, y_val))

    train_list_xx = [df_train_x_ori]
    train_list_x = [df_Original_xtrain]
    train_list_yy = [df_train_y_ori]
    train_list_y = [df_Original_ytrain]

    folder_path = ['Original']
    for j in range(0, len(train_list_xx)):
        xx_train = train_list_xx[j]
        x_train = train_list_x[j]
        yy_train = train_list_yy[j]
        y_train = train_list_y[j]
        method = folder_path[j]
        y_SVR_tr, y_SVR_te, C, gamma, epsilon, kernel = svr_modeling(xx_train, x_val, x_train, x_test, yy_train, y_val, y_train, method, iter)
        print("SVR" + str(iter) + "_success____________________")

        y_XGB_tr, y_XGB_te, NE_X, MD_X, SS_X, CB_X, MC_X, LR_X = xgb_modeling(xx_train, x_val, x_train, x_test, yy_train, y_val, y_train, method, iter)
        print("XGB" + str(iter) + "_success____________________")

        input_size = x_train.shape[1]
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, xx_train, x_val, yy_train, y_val, input_size), n_trials=100)
        # 최적의 하이퍼파라미터 추출
        best_hyperparams = study.best_trial.params
        best_hidden_size = best_hyperparams['hidden_size']
        best_learning_rate = best_hyperparams['learning_rate']
        best_hidden_af = best_hyperparams['hidden_activation']
        best_output_af = best_hyperparams['output_activation']
        model = ANN(input_size, best_hidden_size, 1, best_hidden_af, best_output_af)
        criterion = nn.MSELoss()  # L1Loss(절대오차의 평균), SmoothL1iLoss (MSELoss, L1Loss조합), MSELoss(제곱오차의 평균)
        optimizer = optim.Adam(model.parameters(), lr=best_learning_rate)

        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(torch.from_numpy(x_train).float())
            loss = criterion(outputs, torch.from_numpy(y_train.reshape(-1, 1)).float())
            loss.backward()
            optimizer.step()

        model.eval()
        # 모델 저장
        torch.save(model.state_dict(), os.path.join(os.getcwd(), "temp", "05_ANN", "ANN_CDOM_" + str(method) + "_" + str(iter) + '.pth'))
        with torch.no_grad():
            # 훈련 데이터 예측
            y_ANN_tr = model(torch.from_numpy(x_train).float()).numpy()
            # 테스트 데이터 예측
            y_ANN_te = model(torch.from_numpy(x_test).float()).numpy()
        print("ANN" + str(iter) + "_success____________________")
        del model, optimizer, study, criterion

        y_train_re = (y_train * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
        y_test_re = (y_test * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()

        y_SVR_tr_re = (y_SVR_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
        y_RFR_tr_re = (y_RFR_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
        y_XGB_tr_re = (y_XGB_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
        y_ANN_tr_re = (y_ANN_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
        y_tr_data = pd.DataFrame(np.concatenate([np.array(y_train_re).reshape(len(y_train_re), 1),
                                                 np.array(y_SVR_tr_re).reshape(len(y_SVR_tr_re), 1),
                                                 np.array(y_RFR_tr_re).reshape(len(y_RFR_tr_re), 1),
                                                 np.array(y_XGB_tr_re).reshape(len(y_XGB_tr_re), 1),
                                                 np.array(y_ANN_tr_re).reshape(len(y_ANN_tr_re), 1)
                                                 ], axis=1),
                                 columns=['OBS', 'SVR', 'RFR', 'XGB', 'ANN'])

        y_SVR_te_re = (y_SVR_te * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
        y_RFR_te_re = (y_RFR_te * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
        y_XGB_te_re = (y_XGB_te * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
        y_ANN_te_re = (y_ANN_te * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
        y_te_data = pd.DataFrame(np.concatenate([np.array(y_test_re).reshape(len(y_test_re), 1),
                                                 np.array(y_SVR_te_re).reshape(len(y_SVR_te_re), 1),
                                                 np.array(y_RFR_te_re).reshape(len(y_RFR_te_re), 1),
                                                 np.array(y_XGB_te_re).reshape(len(y_XGB_te_re), 1),
                                                 np.array(y_ANN_te_re).reshape(len(y_ANN_te_re), 1)
                                                 ], axis=1),
                                 columns=['OBS', 'SVR', 'RFR', 'XGB', 'ANN'])

        scatter_plot("Support Vector Machine_" + str(iter), y_train_re, y_SVR_tr_re, y_test_re, y_SVR_te_re, "Observed CDOM ($m{-1}$)",
                     "Simulated CDOM ($m^{-1}$)",
                     os.path.join(os.getcwd(), "graph", "SVR_" + str(iter) + ".png"))
        plt.clf()
        scatter_plot("Random Forest_" + str(iter), y_train_re, y_RFR_tr_re, y_test_re, y_RFR_te_re, "Observed CDOM ($m{-1}$)",
                     "Simulated CDOM ($m^{-1}$)",
                     os.path.join(os.getcwd(), "graph", "RFR_" + str(iter) + ".png"))
        plt.clf()
        scatter_plot("XGBoost_" + str(iter), y_train_re, y_XGB_tr_re, y_test_re, y_XGB_te_re, "Observed CDOM ($m{-1}$)",
                     "Simulated CDOM ($m^{-1}$)",
                     os.path.join(os.getcwd(), "graph", "XGB_" + str(iter) + ".png"))
        plt.clf()
        try:
            scatter_plot("Artificial Neural Network_" + str(iter), y_train_re, y_ANN_tr_re, y_test_re, y_ANN_te_re, "Observed CDOM ($m{-1}$)",
                         "Simulated CDOM ($m^{-1}$)",
                         os.path.join(os.getcwd(), "graph", "ANN_" + str(iter) + ".png"))
            plt.clf()
        except:
            pass


        SVR1, SVR2, SVR3, SVR4, SVR5, SVR6 = accuracy_score(y_train_re, y_test_re, y_SVR_tr_re, y_SVR_te_re)
        RFR1, RFR2, RFR3, RFR4, RFR5, RFR6 = accuracy_score(y_train_re, y_test_re, y_RFR_tr_re, y_RFR_te_re)
        XGB1, XGB2, XGB3, XGB4, XGB5, XGB6 = accuracy_score(y_train_re, y_test_re, y_XGB_tr_re, y_XGB_te_re)
        ANN1, ANN2, ANN3, ANN4, ANN5, ANN6 = accuracy_score(y_train_re, y_test_re, y_ANN_tr_re, y_ANN_te_re)

        result_SVR = open(os.path.join(os.getcwd(), "M01_SVR_model.csv"), "a")
        result_RFR = open(os.path.join(os.getcwd(), "M02_RFR_model.csv"), "a")
        result_XGB = open(os.path.join(os.getcwd(), "M03_XGB_model.csv"), "a")
        result_ANN = open(os.path.join(os.getcwd(), "M04_ANN_model.csv"), "a")

        result_SVR.write(f"{iter},SVR,{method},{SVR1},{SVR2},{SVR3},{SVR4},{SVR5},{SVR6},"
                         f"{C},{gamma},{epsilon},{kernel}\n")

        result_RFR.write(f"{iter},RFR,{method},{RFR1},{RFR2},{RFR3},{RFR4},{RFR5},{RFR6},"
                         f"{NE_R},{MD_R},{MSS_R},{MSL_R},{MF_R}\n")

        result_XGB.write(f"{iter},XGB,{method},{XGB1},{XGB2},{XGB3},{XGB4},{XGB5},{XGB6},"
                         f"{NE_X},{MD_X},{SS_X},{CB_X},{MC_X},{LR_X}\n")

        result_ANN.write(f"{iter},ANN,{method},{ANN1},{ANN2},{ANN3},{ANN4},{ANN5},{ANN6},"
                         f"{best_hidden_size},{best_learning_rate},{best_hidden_af},{best_output_af}\n")

        result_SVR.close()
        result_RFR.close()
        result_XGB.close()
        result_ANN.close()

        with pd.ExcelWriter(
                os.path.join(os.getcwd(), "result", "A_yresult_" + str(iter) + "_" + str(method) + ".xlsx")) \
                as writer:

            y_tr_data.to_excel(writer, sheet_name="train")
            y_te_data.to_excel(writer, sheet_name="test")
            tr_df = pd.DataFrame(
                np.concatenate(
                    (np.array(y_train_re).reshape(len(y_train_re), 1), x_train),
                    axis=1))
            tr_df.to_excel(writer, sheet_name='train_data')
            test_df = pd.DataFrame(
                np.concatenate((np.array(y_test_re).reshape(len(y_test_re), 1), x_test), axis=1))
            test_df.to_excel(writer, sheet_name='test_data')
        del y_tr_data, y_te_data, tr_df, test_df
    del df_tr_nor, df_vl_nor, df_te_nor
    iter += 1
