import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from util.metric import *
import matplotlib.pyplot as plt

"""
Classical models:
    * AR:       Autoregressive
    * MA:       Moving Average
    * ARIMA:    Autoregressive integrated and moving average
    * ES:       Exponential Smoothing
    * LR:       Linear Regression
    * SVR:      Support Vector Regression (Kernel: RGB, linear, gaussian, sigmoid)
    * LASSO:    LASSO
    * KNN:      K-Nearest Neighbors
    * RF:       Random Forest Regression
"""


def AR(data=None, params=None):
    """
    Build AR model => Train model, find best p to minimize BIC => Get forecasts.
    :param data: history data.
    :param params: parameters.
    :return: 1-d array, Forecasts in next h time step.
    """
    model = auto_arima(y=data, start_p=params['startp'], start_q=0, max_p=params['end_q'], max_q=0)
    fcasts = model.predict(n_periods=params['horizon'])
    return fcasts


def MA(data=None, params=None):
    """
    Build MA model => Train model => Get forecasts.
    :param data: history data.
    :param horizon: Length of forecasts.
    :return: 1-d array, Forecasts in next h time steps.
    """
    model = auto_arima(y=data, start_p=0, start_q=params['startq'], max_p=0, max_q=params['maxq'])
    fcasts = model.predict(n_periods=params['horizon'])
    return fcasts


def ARIMA(data=None, params=None):
    """
    Build ARIMA model => Train model => Get forecasts.
    :param data: history data.
    :param params: parameters.
    :return: 1-d array, Forecasts in next h time steps.
    """
    model = auto_arima(y=data, max_p=params['maxp'], max_q=params['maxq'], start_q=params['startq'], start_p=params['startp'], n_jobs=-1)
    model.fit(data)
    fcasts = model.predict(n_periods=params['horzion'])
    return fcasts,model


def ES(data=None, horizon=24, alpha=0.3):
    """
    Build ES model => Train model => Get forecasts.
    :param data: history data.
    :param horizon: Length of forecasts.
    :return: list, Forecasts in next h time steps.
    """
    model = SimpleExpSmoothing(data).fit(smoothing_level=alpha)
    fcasts = model.predict(start=len(data), end=len(data)+horizon)
    return fcasts


def gen_data_1d(ts=None, history=24, horizon=12):
    """
    Generate X, Y for regression models.
    :param ts: 1-d time series
    :param history:
    :return: X, Y
    """
    T = ts.shape[0]
    n_sample = T - history - horizon + 1
    X = np.zeros((n_sample, history))
    Y = np.zeros((n_sample, horizon))
    for i in range(n_sample):
        X[i, :] = ts[i:i+history]
        Y[i, :] = ts[i+history:i+history+horizon]
    return X, Y


def KNN_reg(data=None, param=None):
    """
    KNN regression for time series. Using 10% samples as validation data, 10% as test data.
    :param data: dictionary, data['train'], data['valid']
    :param param: dictionary.
    :return: best model on validation set.
    """
    min_error = np.inf
    best_model = None

    mse_list = []
    k_list = []

    for k in range(param['k_start'], param['k_end']+1):
        k_list.append(k)
        model = KNeighborsRegressor(n_neighbors=k, weights=param['weight'], metric=param['dist'])
        model.fit(data['train'][0], data['train'][1])
        fcasts = one_step_predict(model, data['valid'][0])
        err = np.mean((fcasts - data['valid'][1])**2)
        mse_list.append(err)
        if err < min_error:
            min_error = err
            best_model = model

    return best_model


def one_step_predict(model=None, input=None):
    """
    Predict next value.
    :param model: models(KNN)
    :param input: [n_sample, history]
    :return: predictions [[n_sample]]
    """
    predictions = model.predict(input)
    return predictions

def multi_step_predict(model=None, input=None, horizon=12, scaler=None):
    """
    Predict next horizon values.
    :param model: models(KNN)
    :param input: [n_sample, history]
    :param horizon: horizon
    :return: predictions [n_sample, horizon]
    """
    input = input.copy()
    n_sample = input.shape[0]
    predictions = np.zeros((n_sample, horizon))
    if scaler is not None:
        s_min = scaler.data_min_[-1]
        s_max = scaler.data_max_[-1]
    else:
        s_min = 0
        s_max = 1
    for h in range(horizon):
        pred = model.predict(input)
        predictions[:, h] = pred.reshape(1, -1)
        input[:, :-1] = input[:, 1:]
        if s_max - s_min != 0 :
            pred = (pred - s_min) / (s_max - s_min)
        else:
            pred = pred
        input[:, -1] = pred.reshape(1, -1)
    return predictions


def regression(data=None, params=None):
    if params['model'] == 'SVR':
        model = svm.SVR(kernel=params['kernel'])
    elif params['model'] == 'Ridge':
        model = linear_model.Ridge(alpha=params['alpha'])
    elif params['model'] == 'LASSO':
        model = linear_model.Lasso(alpha=params['alpha'])
    elif params['model'] == 'RF':
        model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
    elif params['model'] == 'LR':
        model = linear_model.LinearRegression()
    model.fit(data['train'][0], data['train'][1].ravel())
    return model

