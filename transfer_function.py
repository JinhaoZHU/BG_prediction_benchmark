# from util.load_data import *
from util.pytorchtools import *

from models.ClassicalModels import *
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import os
import joblib

import argparse
import time
from util.data_utils import *
from util.optim import *
from models.DeepModels import *
from util.metric import *
from util.pytorchtools import *
import torch
from torch import nn
import os


def multi_step_transfer(data_type, patient, model, model_path, history_minutes=150, horizon=1, suffix='csv', epochs=300,
                        is_train=True, detailed=True, use_all=False, exclusive=False, use_meal=False, normalize=0,
                        use_insulin=False, val_data_type=None):
    # 对未来的n个horizon建立一个模型
    prediction_type = 'multi_step'

    class Dict2Obj(dict):
        def __init__(self, *args, **kwargs):
            super(Dict2Obj, self).__init__(*args, **kwargs)

        def __getattr__(self, key):
            value = self[key]
            if isinstance(value, dict):
                value = Dict2Obj(value)
            return value

    if data_type in interval_15:
        time_interval = 15
    else:
        time_interval = 5

    params = {
        'patient': patient,
        'data': f'data/{data_type}/{patient}.{suffix}',
        'train_data': f'data/{data_type}/{patient}-ws-training.csv',
        'test_data': f'data/{data_type}/{patient}-ws-testing.csv',
        'horizon': 1,
        'output_len': 12,
        'window': history_minutes // time_interval,
        'highway_window': 14,
        'skip': -1,
        'model': f'{model}',
        'CNN_kernel': 2,
        'hidRNN': 512,
        'hidCNN': 50,
        'hidSkip': 0,
        'L1Loss': False,
        'epochs': epochs,
        'batch_size': 512,
        'output_fun': 'linear',
        'dropout': 0.2,
        # 'save': f'save/{prediction_type}/{data_type}/{patient}/{model}.pt',
        'save': f'save/{model_path}/{data_type}/{model}_transfer_{val_data_type}.pt',
        'clip': 10,
        'seed': 12345,
        'log_interval': 2000,
        'optim': 'adam',
        'lr': 0.001,
        'normalize': normalize,
        'gpu': 0,
        'cuda': 1,
        'm': 1
    }

    params = Dict2Obj(params)
    print(os.path.dirname(params.save))
    if not os.path.exists(os.path.dirname(params.save)):
        os.makedirs(os.path.dirname(params.save))
    if history_minutes >= 700:
        params['batch_size'] = 256
        # if history_minutes >= 1000:
        #     params['batch_size'] = 1

    @torch.no_grad()
    def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, ifsave=False, ds='ds'):
        model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        for X, Y in data.get_batches(X, Y, batch_size, False):
            output = model(X)
            scale = data.scale[0].expand(Y.size(0), Y.size(1))
            total_loss += evaluateL2(output * scale, Y * scale).item()
            total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
            n_samples += (output.size(0) * data.m)

            if predict is None:
                predict = output * scale
                test = Y * scale
            else:
                predict = torch.cat((predict, output * scale))
                test = torch.cat((test, Y * scale))

        rse = math.sqrt(total_loss / n_samples) / data.rse
        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        v = Ytest.reshape(-1)
        v_ = predict.reshape(-1)
        mae = MAE(v, v_)
        mape = MAPE(v, v_)
        rmse = RMSE(v, v_)

        return rse.item(), rmse, mape, mae

    @torch.no_grad()
    def multioutput_eval(data, X, Y, model, evaluateL2, evaluateL1, batch_size, args, save_path=None, detailed=False):

        model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        for X, Y in data.get_batches(X, Y, batch_size, False):
            output = model(X)
            scale = data.scale[0].expand(Y.size(0), Y.size(1))
            total_loss += evaluateL2(output * scale, Y * scale).item()
            total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
            n_samples += (output.size(0) * data.m)

            if predict is None:
                predict = output * scale
                test = Y * scale
            else:
                predict = torch.cat((predict, output * scale))
                test = torch.cat((test, Y * scale))

        n_samples = predict.size(0)
        predict = predict.data.cpu().numpy()
        test = test.data.cpu().numpy()
        l1_loss = []
        rmse_loss = []
        mape_loss = []
        mae_loss = []
        smape_loss = []
        smape_abs_loss = []
        error_in_5_loss = []
        error_in_10_loss = []
        clarke_loss = []
        parkes_loss = []

        for i in range(args.output_len):
            # l1_loss.append(evaluateL1(predict[:,i],test[:,i])/n_samples)
            # rmse_loss.append(math.sqrt(evaluateL2(predict[:,i],test[:,i])/n_samples))
            rmse_loss.append(RMSE(predict[:, i], test[:, i]))
            mape_loss.append(MAPE(predict[:, i], test[:, i]))
            mae_loss.append(MAE(predict[:, i], test[:, i]))
            smape_loss.append(sMAPE(predict[:, i], test[:, i]))
            smape_abs_loss.append(sMAPE_abs(predict[:, i], test[:, i]))
            error_in_5_loss.append(sum(error_in_5(predict[:, i], test[:, i])) / n_samples)
            error_in_10_loss.append(sum(error_in_10(predict[:, i], test[:, i])) / n_samples)
            clarke_loss.append(list(zone_accuracy(test[:, i], predict[:, i], mode='clarke', diabetes_type=1)))
            parkes_loss.append(list(zone_accuracy(test[:, i], predict[:, i], mode='parkes', diabetes_type=1)))

        return rmse_loss, mape_loss, mae_loss, [smape_loss, smape_abs_loss, error_in_5_loss, error_in_10_loss,
                                                clarke_loss, parkes_loss, test, predict, Data.test_time]

    def train(data, X, Y, model, criterion, optim, batch_size):
        model.train()
        total_loss = 0
        n_samples = 0
        for X, Y in data.get_batches(X, Y, batch_size, True):
            model.zero_grad()
            output = model(X)
            scale = data.scale[0].expand(Y.size(0), Y.size(1))
            loss = criterion(output * scale, Y * scale)
            loss.backward()
            grad_norm = optim.step()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
        return total_loss / n_samples

    print(params['output_len'])

    Data = Data_utility(data_type, params['data'], params['train_data'], params['test_data'], 0.6, 0.2, params['cuda'],
                        params['horizon'], params['window'],
                        params['normalize'], output_len=params['output_len'], use_meal=use_meal,
                        use_insulin=use_insulin)

    val_data = Data_utility(data_type, params['data'], params['train_data'], params['test_data'], 0.6, 0.2,
                            params['cuda'],
                            params['horizon'], params['window'],
                            params['normalize'], output_len=params['output_len'], use_meal=use_meal,
                            use_insulin=use_insulin)

    print('获取全部训练数据')
    Data.train, Data.valid, Data.test = get_all_data(data_type, params, exclusive,resam)

    print('获取另一数据集的验证数据')
    val_data.test = get_all_data(val_data_type, params, exclusive)[2]

    model = eval(params['model'])(params, Data)

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    if params['L1Loss']:
        criterion = nn.L1Loss(size_average=False)
    else:
        criterion = nn.MSELoss(size_average=False)
    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()
        model = model.cuda()

    best_val = 10000000
    optim = Optim(
        model.parameters(), params['optim'], params['lr'], params['clip'],
    )
    early_stopping = EarlyStopping(30, verbose=True, path=params['save'] + '.ckp', model_type=model)

    if not os.path.exists(os.path.dirname(params['save'])):
        os.makedirs(os.path.dirname(params['save']))

    # At any point you can hit Ctrl + C to break out of training early.
    all_start = time.time()
    if is_train:
        try:
            print('begin training')
            for epoch in range(1, params['epochs'] + 1):
                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, params['batch_size'])
                val_rse, val_rmse, val_mape, val_mae = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2,
                                                                evaluateL1,
                                                                params['batch_size'])
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rsme {:5.4f} | valid mape  {:5.4f} | valid mae  {:5.4f} '.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_rse, val_rmse, val_mape, val_mae))
                # Save the model if the validation loss is the best we've seen so far.

                if val_rse < best_val:
                    with open(params['save'], 'wb') as f:
                        torch.save(model, f)
                    best_val = val_rse
                if epoch % 5 == 0:
                    test_rse, test_rmse, test_mape, test_mae = evaluate(Data, Data.test[0], Data.test[1], model,
                                                                        evaluateL2,
                                                                        evaluateL1,
                                                                        params['batch_size'])
                    print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae {:5.4f}".format(test_rse,
                                                                                                               test_rmse,
                                                                                                               test_mape,
                                                                                                               test_mae))
                early_stopping(val_rmse, model)
                # 若满足 early stopping 要求
                if early_stopping.early_stop:
                    print("Early stopping")
                    # 结束模型训练
                    break
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
    all_end = time.time()

    # Load the best saved model.
    with open(params['save'], 'rb') as f:
        model = torch.load(f)
    # test_rse, test_rmse, test_mape, test_mae = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
    rmse_loss, mape_loss, mae_loss, detail = multioutput_eval(val_data, val_data.test[0], val_data.test[1], model,
                                                              evaluateL2,
                                                              evaluateL1,
                                                              params['batch_size'], params, detailed=detailed)

    # print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae  {:5.4f}".format(test_rse, test_rmse, test_mape, test_mae))
    return [rmse_loss, mape_loss, mae_loss] + detail


def direct_transfer(data_type, patient, model, model_path, history_minutes=150, horizon=1, suffix='csv', epochs=300,
                        is_train=True, detailed=True, use_all=False, exclusive=False, use_meal=False, normalize=0,
                        use_insulin=False, val_data_type=None):
    # 对未来的n个horizon建立一个模型
    prediction_type = 'multi_step'

    class Dict2Obj(dict):
        def __init__(self, *args, **kwargs):
            super(Dict2Obj, self).__init__(*args, **kwargs)

        def __getattr__(self, key):
            value = self[key]
            if isinstance(value, dict):
                value = Dict2Obj(value)
            return value

    if data_type in interval_15:
        time_interval = 15
    else:
        time_interval = 5

    params = {
        'patient': patient,
        'data': f'data/{data_type}/{patient}.{suffix}',
        'train_data': f'data/{data_type}/{patient}-ws-training.csv',
        'test_data': f'data/{data_type}/{patient}-ws-testing.csv',
        'horizon': horizon,
        'window': history_minutes // time_interval,
        'output_len': 1,
        'highway_window': 14,
        'skip': -1,
        'model': f'{model}',
        'CNN_kernel': 2,
        'hidRNN': 512,
        'hidCNN': 50,
        'hidSkip': 0,
        'L1Loss': False,
        'epochs': epochs,
        'batch_size': 512,
        'output_fun': 'linear',
        'dropout': 0.2,
        'save': f'save/{model_path}/{data_type}/{model}_{horizon}_transfer_{val_data_type}.pt',
        'clip': 10,
        'seed': 12345,
        'log_interval': 2000,
        'optim': 'adam',
        'lr': 0.001,
        'normalize': normalize,
        'gpu': 0,
        'cuda': 1,
        'm': 1
    }

    params = Dict2Obj(params)
    print(os.path.dirname(params.save))
    if not os.path.exists(os.path.dirname(params.save)):
        os.makedirs(os.path.dirname(params.save))
    if history_minutes >= 700:
        params['batch_size'] = 256
        # if history_minutes >= 1000:
        #     params['batch_size'] = 1

    @torch.no_grad()
    def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, ifsave=False, ds='ds'):
        model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None
        for X, Y in data.get_batches(X, Y, batch_size, False):
            output = model(X)
            Y = Y[:, -1:]
            scale = data.scale.expand(output.size(0), data.m)
            total_loss += evaluateL2(output * scale, Y * scale).item()
            total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
            n_samples += (output.size(0) * data.m)

            if predict is None:
                predict = output * scale
                test = Y * scale
            else:
                predict = torch.cat((predict, output * scale))
                test = torch.cat((test, Y * scale))

        rse = math.sqrt(total_loss / n_samples) / data.rse
        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        v = Ytest.reshape(-1)
        v_ = predict.reshape(-1)
        mae = MAE(v, v_)
        mape = MAPE(v, v_)
        rmse = RMSE(v, v_)
        smape = sMAPE(v, v_)
        smape_abs = sMAPE_abs(v, v_)

        error_in_5_loss = sum(error_in_5(v_, v)) / n_samples
        error_in_10_loss = sum(error_in_10(v_, v)) / n_samples
        clarke_loss = list(zone_accuracy(v, v_, mode='clarke', diabetes_type=1))
        parkes_loss = list(zone_accuracy(v, v_, mode='parkes', diabetes_type=1))

        return rse.item(), rmse, mape, mae, [smape, smape_abs, error_in_5_loss, error_in_10_loss, clarke_loss,
                                             parkes_loss, v, v_]

    def train(data, X, Y, model, criterion, optim, batch_size):
        model.train()
        total_loss = 0
        n_samples = 0
        for X, Y in data.get_batches(X, Y, batch_size, True):
            model.zero_grad()
            output = model(X)
            Y = Y[:, -1:]
            scale = data.scale.expand(output.size(0), data.m)
            loss = criterion(output * scale, Y * scale)
            loss.backward()
            grad_norm = optim.step()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
        return total_loss / n_samples

    print(params['output_len'])

    Data = Data_utility(data_type, params['data'], params['train_data'], params['test_data'], 0.6, 0.2, params['cuda'],
                        params['horizon'], params['window'],
                        params['normalize'], output_len=params['output_len'], use_meal=use_meal,
                        use_insulin=use_insulin)

    val_data = Data_utility(data_type, params['data'], params['train_data'], params['test_data'], 0.6, 0.2,
                            params['cuda'],
                            params['horizon'], params['window'],
                            params['normalize'], output_len=params['output_len'], use_meal=use_meal,
                            use_insulin=use_insulin)

    print('获取全部训练数据')
    Data.train, Data.valid, Data.test = get_all_data(data_type, params, exclusive)

    print('获取另一数据集的验证数据')
    val_data.test = get_all_data(val_data_type, params, exclusive)[2]

    model = eval(params['model'])(params, Data)

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    if params['L1Loss']:
        criterion = nn.L1Loss(size_average=False)
    else:
        criterion = nn.MSELoss(size_average=False)
    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()
        model = model.cuda()

    best_val = 10000000
    optim = Optim(
        model.parameters(), params['optim'], params['lr'], params['clip'],
    )
    early_stopping = EarlyStopping(30, verbose=True, path=params['save'] + '.ckp', model_type=model)

    if not os.path.exists(os.path.dirname(params['save'])):
        os.makedirs(os.path.dirname(params['save']))

    # At any point you can hit Ctrl + C to break out of training early.
    all_start = time.time()
    if is_train:
        try:
            print('begin training')
            for epoch in range(1, params['epochs'] + 1):
                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, params['batch_size'])
                val_rse, val_rmse, val_mape, val_mae = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2,
                                                                evaluateL1,
                                                                params['batch_size'])
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rsme {:5.4f} | valid mape  {:5.4f} | valid mae  {:5.4f} '.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_rse, val_rmse, val_mape, val_mae))
                # Save the model if the validation loss is the best we've seen so far.

                if val_rse < best_val:
                    with open(params['save'], 'wb') as f:
                        torch.save(model, f)
                    best_val = val_rse
                if epoch % 5 == 0:
                    test_rse, test_rmse, test_mape, test_mae = evaluate(Data, Data.test[0], Data.test[1], model,
                                                                        evaluateL2,
                                                                        evaluateL1,
                                                                        params['batch_size'])
                    print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae {:5.4f}".format(test_rse,
                                                                                                               test_rmse,
                                                                                                               test_mape,
                                                                                                               test_mae))
                early_stopping(val_rmse, model)
                # 若满足 early stopping 要求
                if early_stopping.early_stop:
                    print("Early stopping")
                    # 结束模型训练
                    break
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
    all_end = time.time()

    # Load the best saved model.
    with open(params['save'], 'rb') as f:
        model = torch.load(f)
    # test_rse, test_rmse, test_mape, test_mae = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
    rmse_loss, mape_loss, mae_loss, detail = evaluate(val_data, val_data.test[0], val_data.test[1], model,
                                                              evaluateL2,
                                                              evaluateL1,
                                                              params['batch_size'], params)

    # print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae  {:5.4f}".format(test_rse, test_rmse, test_mape, test_mae))
    return [rmse_loss, mape_loss, mae_loss] + detail

def recursive_transfer(data_type, patient, model, model_path, history_minutes=150, horizon=12, suffix='csv', epochs=300,
                        is_train=True, detailed=True, use_all=False, exclusive=False, use_meal=False, normalize=0,
                        use_insulin=False, val_data_type=None,downsampling=False):
    # 对未来的n个horizon建立一个模型
    prediction_type = 'recurrent_step'

    class Dict2Obj(dict):
        def __init__(self, *args, **kwargs):
            super(Dict2Obj, self).__init__(*args, **kwargs)

        def __getattr__(self, key):
            value = self[key]
            if isinstance(value, dict):
                value = Dict2Obj(value)
            return value

    if data_type in interval_15:
        time_interval = 15
    else:
        time_interval = 5

    params = {
        'patient':patient,
        'data': f'data/{data_type}/{patient}.{suffix}',
        'train_data': f'data/{data_type}/{patient}-ws-training.csv',
        'test_data': f'data/{data_type}/{patient}-ws-testing.csv',
        'horizon': horizon,
        'window': history_minutes // time_interval,
        'output_len': 1,
        'highway_window': 14,
        'skip': -1,
        'model': f'{model}',
        'CNN_kernel': 2,
        'hidRNN': 512,
        'hidCNN': 50,
        'hidSkip': 0,
        'L1Loss': False,
        'epochs': epochs,
        'batch_size': 512,
        'output_fun': 'linear',
        'dropout': 0.2,
        'save': f'save/{model_path}/{data_type}/{model}_transfer.pt',
        'clip': 10,
        'seed': 12345,
        'log_interval': 2000,
        'optim': 'adam',
        'lr': 0.001,
        'normalize': 0,
        'gpu': 0,
        'cuda': 1
    }

    params = Dict2Obj(params)
    print(os.path.dirname(params.save))
    if not os.path.exists(os.path.dirname(params.save)):
        os.makedirs(os.path.dirname(params.save))

    @torch.no_grad()
    def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, ifsave=False, ds='ds'):
        model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None
        for X, Y in data.get_batches(X, Y, batch_size, False):
            output = model(X)
            Y = Y[:, :1]
            scale = data.scale.expand(output.size(0), data.m)
            total_loss += evaluateL2(output * scale, Y * scale).item()
            total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
            n_samples += (output.size(0) * data.m)

            if predict is None:
                predict = output * scale
                test = Y * scale
            else:
                predict = torch.cat((predict, output * scale))
                test = torch.cat((test, Y * scale))

        rse = math.sqrt(total_loss / n_samples) / data.rse
        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        v = Ytest.reshape(-1)
        v_ = predict.reshape(-1)
        mae = MAE(v, v_)
        mape = MAPE(v, v_)
        rmse = RMSE(v, v_)

        return rse.item(), rmse, mape, mae
    def recurssive_evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, ifsave=False, ds='ds',
                            diabetes_type=1):
        model.eval()
        predict = None
        test = None
        for X, Y in data.get_batches(X, Y, batch_size, False):
            pred = torch.zeros(([X.size(0), data.h]))
            pred = Variable(pred)
            scale0 = data.scale.expand(Y.size(0), data.h)
            for i in range(data.h):
                output = model(X)
                scale = data.scale.expand(output.size(0), data.m)
                X.data[:, :-1, :] = X.data[:, 1:, :]
                X.data[:, -1, :] = output
                pred.data[:, i] = (output * scale).data.reshape(-1)

            if predict is None:
                predict = pred
                test = Y * scale0
            else:
                predict = torch.cat((predict, pred))
                test = torch.cat((test, Y * scale0))

        preds = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        time_step = [i for i in range(params['horizon'])]
        rse = np.zeros(len(time_step))
        rmse = np.zeros(len(time_step))
        mape = np.zeros(len(time_step))
        mae = np.zeros(len(time_step))
        smape_loss = np.zeros(len(time_step))
        smape_abs_loss = np.zeros(len(time_step))
        error_in_5_loss = np.zeros(len(time_step))
        error_in_10_loss = np.zeros(len(time_step))
        clarke_loss = np.zeros((len(time_step), 5))
        parkes_loss = np.zeros((len(time_step), 5))

        for i, t in enumerate(time_step):
            v = Ytest[:, t].reshape(-1)
            v_ = preds[:, t].reshape(-1)
            # rse[i] = RSE(v, v_)
            rmse[i] = RMSE(v, v_)
            mae[i] = MAE(v, v_)
            mape[i] = MAPE(v, v_)
            smape_loss[i] = sMAPE(v_, v)
            smape_abs_loss[i] = sMAPE_abs(v_, v)
            error_in_5_loss[i] = sum(error_in_5(v_, v)) / len(v)
            error_in_10_loss[i] = sum(error_in_10(v_, v)) / len(v)
            clarke_loss[i, :] = list(zone_accuracy(v, v_, mode='clarke', diabetes_type=diabetes_type))
            parkes_loss[i, :] = list(zone_accuracy(v, v_, mode='parkes', diabetes_type=diabetes_type))

        # print(rmse)
        return rmse, mape, mae, [smape_loss, smape_abs_loss, error_in_5_loss, error_in_10_loss, clarke_loss,
                                 parkes_loss, Ytest, preds]

    def train(data, X, Y, model, criterion, optim, batch_size):
        model.train()
        total_loss = 0
        n_samples = 0
        for X, Y in data.get_batches(X, Y, batch_size, True):
            model.zero_grad()
            output = model(X)
            Y = Y[:, :1]
            scale = data.scale.expand(output.size(0), data.m)
            loss = criterion(output * scale, Y * scale)
            loss.backward()
            grad_norm = optim.step()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
        return total_loss / n_samples

    print(params['output_len'])
    
    Data = Data_utility(data_type, params['data'], params['train_data'], params['test_data'], 0.6, 0.2, params['cuda'],
                        params['horizon'], params['window'],
                        params['normalize'], output_len=params['output_len'], use_meal=use_meal,
                        use_insulin=use_insulin,downsampling=downsampling)

    val_data = Data_utility(data_type, params['data'], params['train_data'], params['test_data'], 0.6, 0.2,
                            params['cuda'],
                            params['horizon'], params['window'],
                            params['normalize'], output_len=params['output_len'], use_meal=use_meal,
                            use_insulin=use_insulin,downsampling=downsampling)
    resampling=False
    if data_type in ['simulator_data','ohio_data'] and val_data_type in ['EastT1DM','hospital_data']:
        resampling=True
        # params['save']=params['save']+'_resample'
    elif val_data_type in ['simulator_data','ohio_data'] and data_type in ['EastT1DM','hospital_data']:
        resampling=True
    print('获取全部训练数据')
    Data.train, Data.valid, Data.test = get_all_data(data_type, params, exclusive,resampling=resampling,downsampling=downsampling)

    print('获取另一数据集的验证数据')
    # val_data.test = get_all_data(val_data_type, params, exclusive,resampling=resampling)[2]

    model = eval(params['model'])(params, Data)

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    if params['L1Loss']:
        criterion = nn.L1Loss(size_average=False)
    else:
        criterion = nn.MSELoss(size_average=False)
    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()
        model = model.cuda()

    best_val = 10000000
    optim = Optim(
        model.parameters(), params['optim'], params['lr'], params['clip'],
    )
    early_stopping = EarlyStopping(30, verbose=True, path=params['save'] + '.ckp', model_type=model)

    if not os.path.exists(os.path.dirname(params['save'])):
        os.makedirs(os.path.dirname(params['save']))

    # At any point you can hit Ctrl + C to break out of training early.
    all_start = time.time()
    if is_train:
        try:
            print('begin training')
            for epoch in range(1, params['epochs'] + 1):
                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, params['batch_size'])
                val_rse, val_rmse, val_mape, val_mae = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2,
                                                                evaluateL1,
                                                                params['batch_size'])
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rsme {:5.4f} | valid mape  {:5.4f} | valid mae  {:5.4f} '.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_rse, val_rmse, val_mape, val_mae))
                # Save the model if the validation loss is the best we've seen so far.

                if val_rse < best_val:
                    with open(params['save'], 'wb') as f:
                        torch.save(model, f)
                    best_val = val_rse
                if epoch % 5 == 0:
                    test_rse, test_rmse, test_mape, test_mae = evaluate(Data, Data.test[0], Data.test[1], model,
                                                                        evaluateL2,
                                                                        evaluateL1,
                                                                        params['batch_size'])
                    print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae {:5.4f}".format(test_rse,
                                                                                                               test_rmse,
                                                                                                               test_mape,
                                                                                                               test_mae))
                early_stopping(val_rmse, model)
                # 若满足 early stopping 要求
                if early_stopping.early_stop:
                    print("Early stopping")
                    # 结束模型训练
                    break
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
    all_end = time.time()

    # Load the best saved model.
    with open(params['save'], 'rb') as f:
        print(params['save'])
        model = torch.load(f)

    res=None
    if val_data_type == 'ohio_data':
        patient_list = ohio_list
    else:
        patient_list = os.listdir( f'data/{val_data_type}')
    for patient in patient_list:
        if type(patient) == int:
            suffix = 'csv'
        else:
            patient, suffix = patient.split('.')
        print(f'data/{val_data_type}/{patient}.{suffix}')
        val_data = Data_utility(val_data_type, f'data/{val_data_type}/{patient}.{suffix}', f'data/{val_data_type}/{patient}-ws-training.csv', f'data/{val_data_type}/{patient}-ws-testing.csv', 0.6, 0.2,
                                params['cuda'],
                                params['horizon'], params['window'],
                                params['normalize'], output_len=params['output_len'], use_meal=use_meal,
                                use_insulin=use_insulin,downsampling=downsampling)
        rmse_loss, mape_loss, mae_loss, detail = recurssive_evaluate(val_data, val_data.test[0], val_data.test[1], model,
                                                              evaluateL2,
                                                              evaluateL1,
                                                              params['batch_size'], params)
        if res is None:
            temp=[rmse_loss, mape_loss, mae_loss] + detail
            temp=[np.expand_dims(i,0) for i in temp][:-2]
            res=temp
        else:
            temp=[rmse_loss, mape_loss, mae_loss] + detail
            temp=[np.expand_dims(i,0) for i in temp][:-2]
            for i in range(len(res)):
                res[i]=np.concatenate((res[i],temp[i]),axis=0)
    # print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae  {:5.4f}".format(test_rse, test_rmse, test_mape, test_mae))
    # return [rmse_loss, mape_loss, mae_loss] + detail
    return [np.mean(i,axis=0) for i in res]