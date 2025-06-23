# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Sampling
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
# ANN model
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras
from tensorflow.keras import layers
import timeit
from keras.models import load_model
# ANN
import torch
import torch.nn as nn
import torch.optim as optim
import functools
# GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# SVR
from sklearn.svm import SVR
# RFR
from sklearn.ensemble import RandomForestRegressor
# XGB
import lightgbm as lgb
from xgboost import XGBRegressor
# TabNetRegerssor
from pytorch_tabnet.tab_model import TabNetRegressor
# model save
import pickle
from pickle import dump, load
import joblib
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
# Sampling
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE
# matplotlib
from numpy.polynomial.polynomial import polyfit
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn.model_selection import StratifiedShuffleSplit
import resreg
import smogn


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
    ax.scatter(obs_tr, sim_tr, color='teal', edgecolors='black')
    ax.scatter(obs_te, sim_te, color='sandybrown', edgecolors='black')
    # ax.plot(obs, poly1d(obs), color="red", linestyle="solid", linewidth=0.5, label="fit")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    legend_handle1 = mpatches.Patch(color='teal')  # 표식의 색깔 설정
    legend_handle2 = mpatches.Patch(color='sandybrown')
    ax.legend(handles=[legend_handle1, legend_handle2], labels=['train', 'test'], loc=2)
    poly_1 = str(poly1d1).replace('\n', '')
    poly_2 = str(poly1d2).replace('\n', '')
    ax.add_artist(AnchoredText(f"Train $R^2$: {round(pearson1, 3)}\ny={poly_1}\n"
                               f"Test $R^2$: {round(pearson2, 3)}\ny={poly_2}", loc=1, prop=dict(size=10)))

    ax.scatter(obs_tr, sim_tr, s=60, color='teal', edgecolors='black')
    ax.scatter(obs_te, sim_te, s=60, color='sandybrown', edgecolors='black')
    plt.plot(x, y, color='black', linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close('all')


def svr_modeling(xx_train, x_val, x_train, x_test, yy_train, y_val, y_train, method, iter):
    def objective_svr(trial):
        C = trial.suggest_float('C', 0.1, 100, log=True)
        gamma = trial.suggest_float('gamma', 0.01, 1, log=True)
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
        # rmse = vl_rmse
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
    del objective_svr, study_svr
    return y_SVR_tr, y_SVR_te, C, gamma, epsilon, kernel


def rfr_modeling(xx_train, x_val, x_train, x_test, yy_train, y_val, y_train, method, iter):
    def objective_rfr(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 3, 8)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 3, 5)
        max_features = trial.suggest_float('max_features', 0.75, 0.85)

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
        # rmse = (tr_rmse + vl_rmse) / 2
        rmse = vl_rmse
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
    del objective_rfr, study_rfr
    return y_RFR_tr, y_RFR_te, NE_R, MD_R, MSS_R, MSL_R, MF_R


def xgb_modeling(xx_train, x_val, x_train, x_test, yy_train, y_val, y_train, method, iter):
    def objective_xgb(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 3, 6)
        subsample = trial.suggest_float('subsample', 0.8, 0.9)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 0.8)
        min_child_weight = trial.suggest_int('min_child_weight', 3, 6)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.05, log=True)  # = eta와 같은값

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

        # rmse = (tr_rmse + vl_rmse) / 2
        rmse = vl_rmse
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
    del objective_xgb, study_xgb
    return y_XGB_tr, y_XGB_te, NE_X, MD_X, SS_X, CB_X, MC_X, LR_X

# dnn정의
class DNN(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_units, output_size, hidden_activation, output_activation):
        super(DNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_units))
        self.hidden_activation = self.get_activation(hidden_activation)
        self.output_activation = self.get_activation(output_activation)
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
        self.layers.append(nn.Linear(hidden_units, output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.hidden_activation(layer(x))
        x = self.output_activation(self.layers[-1](x))
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
        elif activation == 'linear':
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

# ANN 모델링 함수
def objective_ann(trial, xx_train, x_val, yy_train, y_val, input_size):
    # 하이퍼파라미터 제안
    hidden_layers = trial.suggest_int('hidden_layers', 1, 5)
    hidden_units = trial.suggest_int('hidden_units', 10, 200)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    hidden_activation = trial.suggest_categorical('hidden_activation', ['relu', 'leaky_relu', 'linear'])
    output_activation = trial.suggest_categorical('output_activation', ['leaky_relu', 'ELU', 'PReLU', 'SELU', 'GELU', 'linear'])

    # 모델 생성
    model = DNN(input_size, hidden_layers, hidden_units, 1, hidden_activation, output_activation).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 루프
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.from_numpy(xx_train).float().to(device))
        loss = criterion(outputs, torch.from_numpy(yy_train.reshape(-1, 1)).float().to(device))
        loss.backward()
        optimizer.step()

    # 검증 데이터 예측 및 평가
    model.eval()
    with torch.no_grad():
        y_tr_pred = model(torch.from_numpy(xx_train).float().to(device)).cpu().numpy()
        y_val_pred = model(torch.from_numpy(x_val).float().to(device)).cpu().numpy()
        tr_rmse = math.sqrt(mean_squared_error(yy_train, y_tr_pred))
        vl_rmse = math.sqrt(mean_squared_error(y_val, y_val_pred))
        rmse = (tr_rmse + vl_rmse) / 2
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


def stratified_split(X, y, test_size=0.25, random_state=None):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, val_idx in splitter.split(X, y >= 5/15):  # y >= 5 조건에 따라 분할
        split_xtrain, split_xval = X[train_idx], X[val_idx]
        split_ytrain, split_yval = y[train_idx], y[val_idx]
    return split_xtrain, split_xval, split_ytrain, split_yval

if __name__ == "__main__":
    # data loading
    # GPR - 159, SVR - 134, RFR/XGB - 198, ANN - 60, TNR - 191
    ML = ['SVR', 'RFR', 'XGB', 'ANN']
    number = [134, 171, 189, 60]

    for jj in range(0, len(ML)):
        inputpath = os.getcwd()
        xlsx_name = "A_yresult_" + str(number[jj]) + "_Original.xlsx"
        df_train_dataset = pd.read_excel(os.path.join(inputpath, xlsx_name), sheet_name='train_data', header=0)
        df_validation_dataset = pd.read_excel(os.path.join(inputpath, xlsx_name), sheet_name='validation_data', header=0)
        df_test_dataset = pd.read_excel(os.path.join(inputpath, xlsx_name), sheet_name='test_data', header=0)

        headers = df_train_dataset.drop(['Level'], axis=1).columns
        headers = list(headers)

        # result = open(os.path.join(inputpath, "M0" + str(jj + 1) + "_" + ML[jj] + "_model.csv"), "w")
        # result.write("iteration,ML,method,train_r2,test_r2,train_MAE,test_MAE,train_RMSE,test_RMSE\n")
        # result.close()

        # Input data normalization
        ymaxmin = [15, 0]
        df_tr_x_ori = df_train_dataset.drop(['a355', 'Level'], axis=1)
        df_tr_x_ori = df_tr_x_ori.values
        df_tr_y_ori = df_train_dataset[['a355']]
        df_tr_y_ori = df_tr_y_ori.values
        df_tr_y_ori = np.array(df_tr_y_ori).reshape(len(df_tr_y_ori))

        df_vl_x_ori = df_validation_dataset.drop(['a355', 'Level'], axis=1)
        x_val = df_vl_x_ori.values
        df_vl_y_ori = df_validation_dataset[['a355']]
        df_vl_y_ori = df_vl_y_ori.values
        y_val = np.array(df_vl_y_ori).reshape(len(df_vl_y_ori))

        df_te_x_ori = df_test_dataset.drop(['a355', 'Level'], axis=1)
        x_test = df_te_x_ori.values
        df_te_y_ori = df_test_dataset[['a355']]
        df_te_y_ori = df_te_y_ori.values
        y_test = np.array(df_te_y_ori).reshape(len(df_te_y_ori))

        # data resampling
        # original training dataset
        df_Original_xtrain = np.vstack((df_tr_x_ori, x_val))
        df_Original_ytrain = np.concatenate((df_tr_y_ori, y_val))

        iter = 1
        while iter < 201:
            # random over-sampling(RO)
            relevance = resreg.sigmoid_relevance(df_tr_y_ori, cl=None, ch=1/3)
            df_train_x_RO, df_train_y_RO = resreg.random_oversample(df_tr_x_ori, df_tr_y_ori, relevance,
                                                                    relevance_threshold=0.3,
                                                                    over='balance',
                                                                    random_state=0)  # under = ['balance', 'average', 'extreme']
            df_RandomOver_xtrain = np.vstack((df_train_x_RO, x_val))
            df_RandomOver_ytrain = np.concatenate((df_train_y_RO, y_val))

            # SmoteR
            df_train_x_ST, df_train_y_ST = resreg.smoter(df_tr_x_ori, df_tr_y_ori, relevance, relevance_threshold=0.3,
                                                         k=5, over='balance', random_state=0)
            df_SmoteR_xtrain = np.vstack((df_train_x_ST, x_val))
            df_SmoteR_ytrain = np.concatenate((df_train_y_ST, y_val))

            # Gaussian noise
            df_train_x_GN, df_train_y_GN = resreg.gaussian_noise(df_tr_x_ori, df_tr_y_ori, relevance,
                                                                 relevance_threshold=0.3,
                                                                 delta=0.1, over='balance',
                                                                 random_state=0)  # delta = [0.01, 0.1, 0.5]
            df_GaussianNoise_xtrain = np.vstack((df_train_x_GN, x_val))
            df_GaussianNoise_ytrain = np.concatenate((df_train_y_GN, y_val))

            # WERCS
            df_train_x_WE, df_train_y_WE = resreg.wercs(df_tr_x_ori, df_tr_y_ori, relevance, over=0.5, under=0.75,
                                                        noise=False,
                                                        delta=0.1, random_state=0)
            df_WERCS_xtrain = np.vstack((df_train_x_WE, x_val))
            df_WERCS_ytrain = np.concatenate((df_train_y_WE, y_val))

            train_list_xx = [df_tr_x_ori, df_train_x_RO, df_train_x_ST, df_train_x_GN, df_train_x_WE]
            train_list_x = [df_Original_xtrain, df_RandomOver_xtrain, df_SmoteR_xtrain, df_GaussianNoise_xtrain,
                            df_WERCS_xtrain]
            train_list_yy = [df_tr_y_ori, df_train_y_RO, df_train_y_ST, df_train_y_GN, df_train_y_WE]
            train_list_y = [df_Original_ytrain, df_RandomOver_ytrain, df_SmoteR_ytrain, df_GaussianNoise_ytrain,
                            df_WERCS_ytrain]

            folder_path = ['Original', 'RandomOver', 'SmoteR', 'GaussianNoise', 'WERCS']
            # folder_path = ['Original']
            for j in range(0, len(train_list_xx)):
                xx_train = train_list_xx[j]
                x_train = train_list_x[j]
                yy_train = train_list_yy[j]
                y_train = train_list_y[j]
                method = folder_path[j]
                y_train_re = (y_train * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
                y_test_re = (y_test * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()

                ##################### GPR ################################################################################
                if ML[jj] == "SVR":
                    y_SVR_tr, y_SVR_te, SV_C, gamma, epsilon, SV_kernel = svr_modeling(xx_train, x_val, x_train, x_test,
                                                                                       yy_train, y_val,
                                                                                       y_train, method, iter)
                    print("SVR" + str(iter) + "_success____________________")

                    y_SVR_tr_re = (y_SVR_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
                    y_tr_data = pd.DataFrame(np.concatenate([np.array(y_train_re).reshape(len(y_train_re), 1),
                                                             np.array(y_SVR_tr_re).reshape(len(y_SVR_tr_re), 1)
                                                             ], axis=1), columns=['OBS', 'SVR'])

                    y_SVR_te_re = (y_SVR_te * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
                    y_te_data = pd.DataFrame(np.concatenate([np.array(y_test_re).reshape(len(y_test_re), 1),
                                                             np.array(y_SVR_te_re).reshape(len(y_SVR_te_re), 1),
                                                             ], axis=1), columns=['OBS', 'SVR'])

                    try:
                        scatter_plot("SVR_" + str(iter), y_train_re, y_SVR_tr_re, y_test_re, y_SVR_te_re,
                                     "Observed CDOM ($m{-1}$)",
                                     "Simulated CDOM ($m^{-1}$)",
                                     os.path.join(os.getcwd(), "graph",
                                                  "2SVR_" + str(method) + "_" + str(iter) + ".png"))
                        plt.clf()
                    except:
                        pass

                    SVR1, SVR2, SVR3, SVR4, SVR5, SVR6 = accuracy_score(y_train_re, y_test_re, y_SVR_tr_re, y_SVR_te_re)

                    result = open(os.path.join(inputpath, "M0" + str(jj + 1) + "_" + ML[jj] + "_model.csv"), "a")
                    result.write(f"{iter},SVR,{method},{SVR1},{SVR2},{SVR3},{SVR4},{SVR5},{SVR6},"
                                 f"{SV_C},{gamma},{epsilon},{SV_kernel}\n")
                    result.close()

                    with pd.ExcelWriter(
                            os.path.join(inputpath, "result", "1B_yresult_" + str(ML[jj]) + "_" + str(iter) + "_"
                                                              + str(method) + ".xlsx")) \
                            as writer:

                        y_tr_data.to_excel(writer, sheet_name="train")
                        y_te_data.to_excel(writer, sheet_name="test")
                        tr_df = pd.DataFrame(
                            np.concatenate(
                                (np.array(y_train_re).reshape(len(y_train_re), 1), x_train),
                                axis=1))
                        tr_df.to_excel(writer, sheet_name='train_data', header=headers)
                        test_df = pd.DataFrame(
                            np.concatenate((np.array(y_test_re).reshape(len(y_test_re), 1), x_test), axis=1))
                        test_df.to_excel(writer, sheet_name='test_data', header=headers)
                    del tr_df, test_df, y_SVR_tr_re, y_SVR_te_re

                ##################### RFR ################################################################################
                elif ML[jj] == "RFR":
                    y_RFR_tr, y_RFR_te, NE_R, MD_R, MSS_R, MSL_R, MF_R = rfr_modeling(xx_train, x_val, x_train, x_test,
                                                                                      yy_train, y_val, y_train, method,
                                                                                      iter)
                    print("RFR" + str(iter) + "_success____________________")
                    y_RFR_tr_re = (y_RFR_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()

                    y_tr_data = pd.DataFrame(np.concatenate([np.array(y_train_re).reshape(len(y_train_re), 1),
                                                             np.array(y_RFR_tr_re).reshape(len(y_RFR_tr_re), 1)
                                                             ], axis=1), columns=['OBS', 'RFR'])

                    y_RFR_te_re = (y_RFR_te * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
                    y_te_data = pd.DataFrame(np.concatenate([np.array(y_test_re).reshape(len(y_test_re), 1),
                                                             np.array(y_RFR_te_re).reshape(len(y_RFR_te_re), 1),
                                                             ], axis=1), columns=['OBS', 'RFR'])

                    scatter_plot("RFR_" + str(iter), y_train_re, y_RFR_tr_re, y_test_re, y_RFR_te_re,
                                 "Observed CDOM ($m{-1}$)",
                                 "Simulated CDOM ($m^{-1}$)",
                                 os.path.join(os.getcwd(), "graph", "3RFR_" + str(method) + "_" + str(iter) + ".png"))
                    plt.clf()

                    RFR1, RFR2, RFR3, RFR4, RFR5, RFR6 = accuracy_score(y_train_re, y_test_re, y_RFR_tr_re, y_RFR_te_re)

                    result = open(os.path.join(inputpath, "M0" + str(jj + 1) + "_" + ML[jj] + "_model.csv"), "a")
                    result.write(f"{iter},RFR,{method},{RFR1},{RFR2},{RFR3},{RFR4},{RFR5},{RFR6},"
                                 f"{NE_R},{MD_R},{MSS_R},{MSL_R},{MF_R}\n")
                    result.close()

                    with pd.ExcelWriter(
                            os.path.join(inputpath, "result", "1B_yresult_" + str(ML[jj]) + "_" + str(iter) + "_"
                                                              + str(method) + ".xlsx")) \
                            as writer:

                        y_tr_data.to_excel(writer, sheet_name="train")
                        y_te_data.to_excel(writer, sheet_name="test")
                        tr_df = pd.DataFrame(
                            np.concatenate(
                                (np.array(y_train_re).reshape(len(y_train_re), 1), x_train),
                                axis=1))
                        tr_df.to_excel(writer, sheet_name='train_data', header=headers)
                        test_df = pd.DataFrame(
                            np.concatenate((np.array(y_test_re).reshape(len(y_test_re), 1), x_test), axis=1))
                        test_df.to_excel(writer, sheet_name='test_data', header=headers)
                    del tr_df, test_df, y_RFR_tr_re, y_RFR_te_re

                ##################### XGB ################################################################################
                elif ML[jj] == "XGB":
                    y_XGB_tr, y_XGB_te, NE_X, MD_X, SS_X, CB_X, MC_X, LR_X = xgb_modeling(xx_train, x_val, x_train,
                                                                                          x_test,
                                                                                          yy_train, y_val, y_train,
                                                                                          method,
                                                                                          iter)
                    print("XGB" + str(iter) + "_success____________________")
                    y_XGB_tr_re = (y_XGB_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()

                    y_tr_data = pd.DataFrame(np.concatenate([np.array(y_train_re).reshape(len(y_train_re), 1),
                                                             np.array(y_XGB_tr_re).reshape(len(y_XGB_tr_re), 1)
                                                             ], axis=1), columns=['OBS', 'XGB'])

                    y_XGB_te_re = (y_XGB_te * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
                    y_te_data = pd.DataFrame(np.concatenate([np.array(y_test_re).reshape(len(y_test_re), 1),
                                                             np.array(y_XGB_te_re).reshape(len(y_XGB_te_re), 1),
                                                             ], axis=1), columns=['OBS', 'XGB'])

                    scatter_plot("XGB_" + str(iter), y_train_re, y_XGB_tr_re, y_test_re, y_XGB_te_re,
                                 "Observed CDOM ($m{-1}$)",
                                 "Simulated CDOM ($m^{-1}$)",
                                 os.path.join(os.getcwd(), "graph", "4XGB_" + str(method) + "_" + str(iter) + ".png"))
                    plt.clf()

                    XGB1, XGB2, XGB3, XGB4, XGB5, XGB6 = accuracy_score(y_train_re, y_test_re, y_XGB_tr_re, y_XGB_te_re)

                    result = open(os.path.join(inputpath, "M0" + str(jj + 1) + "_" + ML[jj] + "_model.csv"), "a")
                    result.write(f"{iter},XGB,{method},{XGB1},{XGB2},{XGB3},{XGB4},{XGB5},{XGB6},"
                                 f"{NE_X},{MD_X},{SS_X},{CB_X},{MC_X},{LR_X}\n")
                    result.close()

                    with pd.ExcelWriter(
                            os.path.join(inputpath, "result", "1B_yresult_" + str(ML[jj]) + "_" + str(iter) + "_"
                                                              + str(method) + ".xlsx")) \
                            as writer:

                        y_tr_data.to_excel(writer, sheet_name="train")
                        y_te_data.to_excel(writer, sheet_name="test")
                        tr_df = pd.DataFrame(
                            np.concatenate(
                                (np.array(y_train_re).reshape(len(y_train_re), 1), x_train),
                                axis=1))
                        tr_df.to_excel(writer, sheet_name='train_data', header=headers)
                        test_df = pd.DataFrame(
                            np.concatenate((np.array(y_test_re).reshape(len(y_test_re), 1), x_test), axis=1))
                        test_df.to_excel(writer, sheet_name='test_data', header=headers)
                    del tr_df, test_df, y_XGB_tr_re, y_XGB_te_re

                ##################### ANN ################################################################################
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    input_size = x_train.shape[1]
                    study_ann = optuna.create_study(direction='minimize')
                    study_ann.optimize(lambda trial: objective_ann(trial, xx_train, x_val, yy_train, y_val, input_size),
                                       n_trials=100)
                    # 최적 하이퍼파라미터
                    best_hyperparams_ann = study_ann.best_trial.params

                    # 최적 모델로 학습
                    best_hidden_layers = best_hyperparams_ann['hidden_layers']
                    best_hidden_units = best_hyperparams_ann['hidden_units']
                    best_hidden_af = best_hyperparams_ann['hidden_activation']
                    best_output_af = best_hyperparams_ann['output_activation']
                    best_learning_rate = best_hyperparams_ann['learning_rate']
                    model = DNN(input_size, best_hidden_layers, best_hidden_units, 1,
                                best_hidden_af, best_output_af).to(device)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=best_learning_rate)

                    for epoch in range(100):
                        model.train()
                        optimizer.zero_grad()
                        outputs = model(torch.from_numpy(x_train).float().to(device))
                        loss = criterion(outputs, torch.from_numpy(y_train.reshape(-1, 1)).float().to(device))
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    # 모델 저장
                    torch.save(model.state_dict(), os.path.join(os.getcwd(), "temp", "05_ANN",
                                                                "ANN_CDOM_" + str(method) + "_" + str(iter) + '.pth'))
                    with torch.no_grad():
                        # 훈련 데이터 예측
                        y_ANN_tr = model(torch.from_numpy(x_train).float().to(device)).cpu().numpy()
                        # 테스트 데이터 예측
                        y_ANN_te = model(torch.from_numpy(x_test).float().to(device)).cpu().numpy()
                    print("ANN" + str(iter) + "_success____________________")
                    del model, optimizer, criterion

                    y_ANN_tr_re = (y_ANN_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()

                    y_tr_data = pd.DataFrame(np.concatenate([np.array(y_train_re).reshape(len(y_train_re), 1),
                                                             np.array(y_ANN_tr_re).reshape(len(y_ANN_tr_re), 1)
                                                             ], axis=1), columns=['OBS', 'ANN'])

                    y_ANN_te_re = (y_ANN_te * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]).flatten()
                    y_te_data = pd.DataFrame(np.concatenate([np.array(y_test_re).reshape(len(y_test_re), 1),
                                                             np.array(y_ANN_te_re).reshape(len(y_ANN_te_re), 1),
                                                             ], axis=1), columns=['OBS', 'ANN'])

                    try:
                        scatter_plot("ANN_" + str(iter), y_train_re, y_ANN_tr_re, y_test_re, y_ANN_te_re,
                                     "Observed CDOM ($m{-1}$)",
                                     "Simulated CDOM ($m^{-1}$)",
                                     os.path.join(os.getcwd(), "graph",
                                                  "5ANN_" + str(method) + "_" + str(iter) + ".png"))
                        plt.clf()
                    except:
                        pass

                    ANN1, ANN2, ANN3, ANN4, ANN5, ANN6 = accuracy_score(y_train_re, y_test_re, y_ANN_tr_re, y_ANN_te_re)

                    result = open(os.path.join(inputpath, "M0" + str(jj + 1) + "_" + ML[jj] + "_model.csv"), "a")
                    result.write(f"{iter},ANN,{method},{ANN1},{ANN2},{ANN3},{ANN4},{ANN5},{ANN6},"
                                 f"{best_hidden_layers},{best_hidden_units},{best_hidden_af},{best_output_af},{best_learning_rate}\n")

                    result.close()

                    with pd.ExcelWriter(
                            os.path.join(inputpath, "result", "1B_yresult_" + str(ML[jj]) + "_" + str(iter) + "_"
                                                              + str(method) + ".xlsx")) \
                            as writer:

                        y_tr_data.to_excel(writer, sheet_name="train")
                        y_te_data.to_excel(writer, sheet_name="test")
                        tr_df = pd.DataFrame(
                            np.concatenate(
                                (np.array(y_train_re).reshape(len(y_train_re), 1), x_train),
                                axis=1))
                        tr_df.to_excel(writer, sheet_name='train_data', header=headers)
                        test_df = pd.DataFrame(
                            np.concatenate((np.array(y_test_re).reshape(len(y_test_re), 1), x_test), axis=1))
                        test_df.to_excel(writer, sheet_name='test_data', header=headers)
                    del tr_df, test_df, y_ANN_tr_re, y_ANN_te_re


            iter += 1