from util.load_data import *
from models.ClassicalModels import *
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import os
import joblib

def onestep_train(modeltype, data_type, patient, time_interval, time_step=[2, 4], suffix='csv', diabetes_type=1):
    # normalize the time_step
    time_step = [i - 1 for i in time_step]

    # Models
    modelset = ['LR', 'SVR', 'Ridge', 'LASSO', 'SVR-polyK', 'RF', 'KNN']
    # modeltype = 'SVR'
    if modeltype not in modelset:
        raise NameError("Don't support this model!")
    params = {'alpha': .2,  # Lasso, Ridge
              'n': 5,  #
              'weight': 'distance',  # KNN: 'uniform', 'distance'
              'dist': 'canberra',
              # KNN: 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mhalanobis'
              'k_start': 12,  # KNN
              'k_end': 15,  # KNN
              'model': modeltype,
              'kernel': 'rbf'}

    history_minutes=150
    history=history_minutes//time_interval
    if time_interval==5:
        horizon = 24
    else:
        horizon=8

    train_size = 0.6
    val_size = 0.2
    test_size = 0.2

    # T=data.shape[0]
    N = 1

    fcasts = np.zeros((0, len(time_step)))
    gts = np.zeros((0, horizon))
    ds = {}
    if data_type == 'ohio_data':
        train_path = f'data/{data_type}/{patient}-ws-training.csv'
        test_path = f'data/{data_type}/{patient}-ws-testing.csv'
        train_data = load_data(train_path, 'csv')['gl_value'].reset_index(drop=True)
        test_data = load_data(test_path, 'csv')['gl_value'].reset_index(drop=True)

        train_data = train_data.interpolate()
        train_data = train_data.dropna(axis=0)

        test_data = test_data.interpolate(method='pad')

        X, Y = gen_data_1d(train_data, history=history, horizon=horizon)
        X_test, Y_test = gen_data_1d(test_data, history=history, horizon=horizon)

        tr = int(X.shape[0] * 0.8)
        va = X.shape[0] - tr - 1
        te = X_test.shape[0]
    else:
        data = load_data(f'data/{data_type}/{patient}.{suffix}', suffix).iloc[:, 0].dropna()

        X, Y = gen_data_1d(data, history=history, horizon=horizon)
        ds = {}
        n_samples = X.shape[0]
        tr = int(n_samples * train_size)
        va = int(n_samples * val_size)
        te = n_samples - tr - va

    fcast = np.zeros((te, len(time_step)))
    for index, i in enumerate(time_step):
        scaler = None
        if data_type == 'ohio_data':
            ds['train'] = [X[:tr, :], Y[:tr, i:i + 1]]
            ds['valid'] = [X[tr:tr + va, :], Y[tr + va, i:i + 1]]
            ds['test'] = [X_test[:, :], Y_test[:, :]]
        else:
            ds['train'] = [X[:tr, :], Y[:tr, i:i + 1]]
            ds['valid'] = [X[tr:tr + va, :], Y[tr + va, i:i + 1]]
            ds['test'] = [X[-te:, :], Y[-te:, :]]
        if modeltype == 'KNN':
            # Normalize
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(X)
            X = scaler.transform(X)
            X[np.isnan(X)] = 0
            model = KNN_reg(data=ds, param=params)

        else:
            # 传统模型只能对单步进行预测，要多步就需要建立函数，不断的预测值放入到输入中，迭代预测
            model = regression(data=ds, params=params)

        input = ds['test'][0].copy()
        n_sample = input.shape[0]
        predictions = np.zeros((n_sample, 1))
        if scaler is not None:
            s_min = scaler.data_min_[-1]
            s_max = scaler.data_max_[-1]
        else:
            s_min = 0
            s_max = 1

        pred = model.predict(input)
        fcast[:, index] = pred.reshape(1, -1)

    # 预测值和ground truth
    fcasts = np.concatenate((fcasts, fcast), axis=0)
    gts = np.concatenate((gts, ds['test'][1]), axis=0)
    # time_test=data[-te-horizon+1:]
    # fcasts = fcasts[:, time_step]
    gts = gts[:, time_step]
    rse = np.zeros((len(time_step)))
    rmse = np.zeros((len(time_step)))
    mape = np.zeros((len(time_step)))
    mae = np.zeros((len(time_step)))
    smape_loss = np.zeros((len(time_step)))
    smape_abs_loss = np.zeros((len(time_step)))
    error_in_5_loss = np.zeros((len(time_step)))
    error_in_10_loss = np.zeros((len(time_step)))
    clarke_loss = np.zeros((len(time_step), 5))
    parkes_loss = np.zeros((len(time_step), 5))
    for i in range(len(time_step)):
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

    print('Dataset:\t', f'result/one_step/{data_type}/{patient}/{modeltype}.csv')
    print("Model:\t", modeltype)
    print("timestep:\t", time_step)
    print("rse:\t", rse)
    print("rmse:\t", rmse)
    print("mape:\t", mape)
    print("mae:\t", mae)
    print("smape:\t", smape_loss)
    print("error_in_5_loss:\t", error_in_5_loss)
    print("error_in_10_loss:\t", error_in_10_loss)
    print("clarke_loss:\t", clarke_loss)
    print("parkes_loss:\t", parkes_loss)

    # save_path = f'result/one_step/{data_type}_{history}/{patient}/{modeltype}.csv'
    save_path = f'result/one_step/{data_type}_{history_minutes}minutes/{patient}/{modeltype}.pkl'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    col = [f'{(i + 1) * time_interval}min' for i in time_step]

    # df = pd.DataFrame(np.vstack([rse, rmse, mape, mae]), index=['rse', 'rmse', 'mape', 'mae'], columns=col)
    # df.to_csv(save_path)
    # res=[rse,rmse,mape,mae,smape_loss,error_in_5_loss,error_in_10_loss,clarke_loss,parkes_loss,gts,fcasts]

    res = [rmse, mape, mae, smape_loss, smape_abs_loss, error_in_5_loss, error_in_10_loss, clarke_loss, parkes_loss,
           gts, fcasts]

    joblib.dump(res, save_path)


def multistep_train(modeltype, data_type, patient, suffix='csv',diabetes_type=1,time_interval=5,downsampling=False):
    # Models
    modelset = ['LR', 'SVR', 'Ridge', 'LASSO', 'SVR-polyK', 'RF', 'KNN']
    # modeltype = 'SVR'
    if modeltype not in modelset:
        raise NameError("Don't support this model!")
    params = {'alpha': .2,  # Lasso, Ridge
              'n': 5,  #
              'weight': 'distance',  # KNN: 'uniform', 'distance'
              'dist': 'canberra',
              # KNN: 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mhalanobis'
              'k_start': 12,  # KNN
              'k_end': 15,  # KNN
              'model': modeltype,
              'kernel': 'rbf'}

    history_minutes=150
    history=history_minutes//time_interval
    if time_interval==5:
        horizon = 12
    else:
        horizon=12
    train_size = 0.6
    val_size = 0.2
    test_size = 0.2

    # T=data.shape[0]
    N = 1


    ds = {}
    if data_type == 'ohio_data':
        train_path = f'data/{data_type}/{patient}-ws-training.csv'
        test_path = f'data/{data_type}/{patient}-ws-testing.csv'
        train_data = load_data(train_path, 'csv')['gl_value']
        train_data.index = pd.to_datetime(train_data.index)
        test_data = load_data(test_path, 'csv')['gl_value']
        test_data.index = pd.to_datetime(test_data.index)

        train_data = train_data.interpolate()
        train_data = train_data.dropna(axis=0)

        test_data = test_data.interpolate(method='pad')
        if downsampling:
            train_data=train_data.resample('15min').first()
            test_data=test_data.resample('15min').first()
            history=history_minutes//15
            horizon=8

        X, Y = gen_data_1d(train_data, history=history, horizon=horizon)
        X_test, Y_test = gen_data_1d(test_data, history=history, horizon=horizon)

        tr = int(X.shape[0] * 0.8)
        va = X.shape[0] - tr - 1
        te = X_test.shape[0]
    else:
        data = load_data(f'data/{data_type}/{patient}.{suffix}', suffix).iloc[:, 0].dropna()
        if downsampling:
            data=data.resample('15min').first()
            history=history_minutes//15
            horizon=8
        X, Y = gen_data_1d(data, history=history, horizon=horizon)
        ds = {}
        n_samples = X.shape[0]
        tr = int(n_samples * train_size)
        va = int(n_samples * val_size)
        te = n_samples - tr - va
    # X, Y=gen_data_1d(data,history=history,horizon=horizon)
    # ds={}
    # n_samples=X.shape[0]
    # tr=int(n_samples*train_size)
    # va=int(n_samples*val_size)
    # te=n_samples-tr-va
    fcasts = np.zeros((0, horizon))
    gts = np.zeros((0, horizon))
    scaler = None
    if data_type == 'ohio_data':
        ds['train'] = [X[:tr, :], Y[:tr, 0:1]]
        ds['valid'] = [X[tr:tr + va, :], Y[tr + va, 0:1]]
        ds['test'] = [X_test[:, :], Y_test[:, :]]
    else:
        ds['train'] = [X[:tr, :], Y[:tr, 0:1]]
        ds['valid'] = [X[tr:tr + va, :], Y[tr + va, 0:1]]
        ds['test'] = [X[-te:, :], Y[-te:, :]]

    if modeltype == 'KNN':
        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X)
        X = scaler.transform(X)
        X[np.isnan(X)] = 0
        model = KNN_reg(data=ds, param=params)

    else:
        model = regression(data=ds, params=params)

    fcast = multi_step_predict(model=model, input=ds['test'][0], horizon=horizon, scaler=scaler)
    fcasts = np.concatenate((fcasts, fcast), axis=0)
    gts = np.concatenate((gts, ds['test'][1]), axis=0)
    time_step = [i for i in range(horizon)]

    fcasts = fcasts[:, time_step]
    gts = gts[:, time_step]
    
    rse = np.zeros((len(time_step)))
    rmse = np.zeros((len(time_step)))
    mape = np.zeros((len(time_step)))
    mae = np.zeros((len(time_step)))
    mae = np.zeros((len(time_step)))
    smape_loss = np.zeros((len(time_step)))
    smape_abs_loss = np.zeros((len(time_step)))
    error_in_5_loss = np.zeros((len(time_step)))
    error_in_10_loss = np.zeros((len(time_step)))
    clarke_loss = np.zeros((len(time_step), 5))
    parkes_loss = np.zeros((len(time_step), 5))
    for i in range(len(time_step)):
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

    print('Dataset:\t', f'result/multi_step/{data_type}_{history_minutes}minutes/{patient}/{modeltype}.pkl')
    print("Model:\t", modeltype)
    print("timestep:\t", time_step)
    print("rse:\t", rse)
    print("rmse:\t", rmse)
    print("mape:\t", mape)
    print("mae:\t", mae)
    print("smape:\t", smape_loss)
    print("error_in_5_loss:\t", error_in_5_loss)
    print("error_in_10_loss:\t", error_in_10_loss)
    print("clarke_loss:\t", clarke_loss)
    print("parkes_loss:\t", parkes_loss)

    if downsampling:
        save_path = f'result/multi_step/{data_type}_{history_minutes}minutes_resample/{patient}/{modeltype}.pkl'
    else:
        save_path = f'result/multi_step/{data_type}_{history_minutes}minutes/{patient}/{modeltype}.pkl'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    res = [rmse, mape, mae, smape_loss, smape_abs_loss, error_in_5_loss, error_in_10_loss, clarke_loss, parkes_loss,
           gts, fcasts]

    joblib.dump(res, save_path)


from util.load_data import *
from models.ClassicalModels import *
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def one_train(data_type, modelset, time_interval=5, time_step=[1, 6, 12, 18], diabetes_type=1):
    # data_type='hospital_data'
    dataset = f'data/{data_type}'
    # if not os.path.exists(os.path.dirname(dataset)):
    #   os.makedirs(os.path.dirname(dataset))

    # data=load_data(dataset,'csv')
    # # T*1 T: length of time series
    # data=data['cgm']
    for patient in os.listdir(dataset):
        print(patient)
        patient, suffix = patient.split('.')
        # data = load_data(dataset + f'/{patient}.{suffix}', suffix)
        # data = data['CGM']
        # modelset = ['LR', 'Ridge', 'LASSO', 'RF', 'KNN']

        for modeltype in tqdm(modelset):
            onestep_train(modeltype, data_type, patient, time_interval, time_step, suffix=suffix,
                          diabetes_type=diabetes_type)
def multi_sumulator(data_type, modelset=['SVR'],diabetes_type=1,time_interval=5,downsampling=False):
    dataset = f'data/{data_type}'
    for patient in os.listdir(dataset):
        print(patient)
        patient, suffix = patient.split('.')

        for modeltype in tqdm(modelset):
            multistep_train(modeltype, data_type, patient, suffix,diabetes_type,time_interval=time_interval,downsampling=downsampling)

if __name__ == '__main__':

    one_train('ShanghaiT1DM', ['LR', 'SVR', 'Ridge', 'LASSO', 'RF', 'KNN'], time_interval=15,
              time_step=[1, 2, 4, 6, 8], diabetes_type=1)
    multi_sumulator('ShanghaiT1DM', ['LR', 'SVR', 'Ridge', 'LASSO', 'RF', 'KNN'],time_interval=5,downsampling=False)