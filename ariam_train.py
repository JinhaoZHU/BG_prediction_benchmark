
from util.load_data import *
from models.ClassicalModels import *
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import random
from classic_model import ohio_list

def arima_train(data_type, patient, time_interval,data, time_step=[2, 4]):
    # Models
    modeltype = 'ARIMA'
    params = {'alpha': .2,  # Lasso, Ridge
              'n': 5,  #
              'weight': 'distance',  # KNN: 'uniform', 'distance'
              'dist': 'canberra',
              # KNN: 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mhalanobis'
              'k_start': 12,  # KNN
              'k_end': 15,  # KNN
              'model': modeltype,
              'kernel': 'linear'}

    history_minutes=150
    history=history_minutes//time_interval
    if time_interval==5:
        horizon = 24
    else:
        horizon=8

    # history = 30
    # horizon = 12

    params['maxp'] = 10
    params['maxq'] = 10
    params['startq'] = 0
    params['startp'] = 0
    params['horzion'] = horizon

    train_size = 0.6
    val_size = 0.2
    test_size = 0.2

    T = data.shape[0]
    N = 1

    fcasts = np.zeros((0, horizon))
    gts = np.zeros((0, horizon))

    X, Y = gen_data_1d(data, history=history, horizon=horizon)
    ds = {}
    n_samples = X.shape[0]
    tr = int(n_samples * train_size)
    va = int(n_samples * val_size)
    te = n_samples - tr - va

    fcast = np.zeros((0, horizon))
    truth=np.zeros((0, horizon))

    i = 1
    index = 0
    scaler = None
    ds['train'] = [X[:tr, :], Y[:tr, i - 1:i]]
    ds['valid'] = [X[tr:tr + va, :], Y[tr + va, i - 1:i]]
    ds['test'] = [X[-te:, :], Y[-te:, :]]

    input = ds['test'][0].copy()
    n_sample = input.shape[0]
    print(f'samples:{n_sample}')
    predictions = np.zeros((n_sample, 1))
    if scaler is not None:
        s_min = scaler.data_min_[-1]
        s_max = scaler.data_max_[-1]
    else:
        s_min = 0
        s_max = 1

    samples=[i for i in range(n_sample)]
    selected=random.sample(samples,min(500,n_sample))
    for i in tqdm(samples):
        x = input[i]
        try:
            y, model = ARIMA(x, params)
        except:
            print(x)
        else:
            fcast=np.concatenate((fcast,np.array(y).reshape((1,-1))),axis=0)
            truth = np.concatenate((truth, np.array(ds['test'][1][i]).reshape((1,-1))), axis=0)
    # pred ,model= ARIMA(input,params)
    # print(pred)
    # fcast[:,index]=pred.reshape(1, -1)

    # 预测值和ground truth
    fcasts = np.concatenate((fcasts, fcast), axis=0)
    # gts = np.concatenate((gts, ds['test'][1]), axis=0)
    gts = np.concatenate((gts, truth), axis=0)

    # fcasts = fcasts[:, time_step]
    # gts = gts[:, 0]
    # fcasts=fcasts[:,0]
    rse = np.zeros((horizon))
    rmse = np.zeros((horizon))
    mape = np.zeros((horizon))
    mae = np.zeros((horizon))
    mae = np.zeros((horizon))
    smape_loss = np.zeros((horizon))
    smape_abs_loss = np.zeros((horizon))
    error_in_5_loss = np.zeros((horizon))
    error_in_10_loss = np.zeros((horizon))
    clarke_loss = np.zeros((horizon, 5))
    parkes_loss = np.zeros((horizon, 5))
    diabetes_type=1
    for i in range(horizon):
        rse[i] = RSE(gts[:, i], fcasts[:, i])
        rmse[i] = RMSE(gts[:, i], fcasts[:, i])
        mape[i] = MAPE(gts[:, i], fcasts[:, i])
        mae[i] = MAE(gts[:, i], fcasts[:, i])

        smape_loss[i] = sMAPE(fcasts[:, i], gts[:, i])
        smape_abs_loss[i] = sMAPE_abs(fcasts[:, i], gts[:, i])
        error_in_5_loss[i] = sum(error_in_5(fcasts[:, i], gts[:, i])) / gts.shape[0]
        error_in_10_loss[i] = sum(error_in_10(fcasts[:, i], gts[:, i])) / gts.shape[0]
        clarke_loss[i] = list(zone_accuracy(gts[:, i], fcasts[:, i], mode='clarke', diabetes_type=diabetes_type))
        parkes_loss[i] = list(zone_accuracy(gts[:, i], fcasts[:, i], mode='parkes', diabetes_type=diabetes_type))

    print('Dataset:\t', f'result/one_step/{data_type}_{history_minutes}minutes/{patient}/{modeltype}.pkl')
    print("Model:\t", modeltype)
    print("rse:\t", rse)
    print("rmse:\t", rmse)
    print("mape:\t", mape)
    print("mae:\t", mae)
    print("smape:\t", smape_loss)
    print("error_in_5_loss:\t", error_in_5_loss)
    print("error_in_10_loss:\t", error_in_10_loss)
    print("clarke_loss:\t", clarke_loss)
    print("parkes_loss:\t", parkes_loss)

    save_path = f'result/one_step/{data_type}_{history_minutes}minutes/{patient}_resample/{modeltype}.pkl'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # col = [f'{(i+1) * time_interval}min' for i in range(horizon)]
    #
    # df = pd.DataFrame(np.vstack([rse, rmse, mape, mae]), index=['rse', 'rmse', 'mape', 'mae'], columns=col)
    # print(save_path)
    # print(df)
    # df.to_csv(save_path)
    res = [rmse, mape, mae, smape_loss, smape_abs_loss, error_in_5_loss, error_in_10_loss, clarke_loss, parkes_loss,
           gts, fcasts]

    joblib.dump(res, save_path)

if __name__ == '__main__':

    data_type = 'ShanghaiT1DM'
    dataset = f'data/{data_type}'
    time_interval = 15

    for patient in os.listdir(dataset):
        print(patient)
        patient, suffix = patient.split('.')
        data = load_data(dataset + f'/{patient}.{suffix}', suffix)
        data = data['cgm'].dropna()
        # modelset = ['LR', 'SVR', 'Ridge', 'LASSO', 'RF', 'KNN']
        arima_train(data_type, patient, time_interval,data, time_step=[1])

