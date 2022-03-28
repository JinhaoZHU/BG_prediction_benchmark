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
from transfer_function import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(1234)


# torch.manual_seed(1234)
# torch.cuda.manual_seed(1234)


def one_step(data_type, patient, model, horizon=1, suffix='csv', model_path="/", is_train=True, time_interval=5,
             history_minutes=60, epochs=300, detailed=True, use_all=False, exclusive=False, use_meal=False, normalize=0,
             use_insulin=False):
    #   对分别horizon分别建立模型
    prediction_type = 'one_step'

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
        'save': f'save/{model_path}/{patient}/{model}_{horizon}.pt',
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

    # if data_type == 'ohio_data':
    #     Data = Ohio_utility(params['train_data'], params['test_data'], 0.6, 0.2, params['cuda'], params['horizon'],
    #                         params['window'], params['normalize'], use_meal=use_meal, use_insulin=use_insulin)
    # else:
    #     Data = Data_utility(params['data'], 0.6, 0.2, params['cuda'], params['horizon'], params['window'],
    #                         params['normalize'], use_meal=use_meal, use_insulin=use_insulin)
    print(params)
    Data = Data_utility(data_type, params['data'], params['train_data'], params['test_data'], 0.6, 0.2, params['cuda'],
                        params['horizon'], params['window'],
                        params['normalize'], output_len=params['output_len'], use_meal=use_meal,
                        use_insulin=use_insulin)

    if use_all:
        print('获取全部数据')
        Data.train = get_all_data(data_type, params, exclusive)[0]
    print(Data.rse)

    model = eval(params['model'])(params, Data).cuda()

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

    best_val = 10000000
    optim = Optim(
        model.parameters(), params['optim'], params['lr'], params['clip'],
    )
    early_stopping = EarlyStopping(30, verbose=True, path=params['save'] + '.ckp')
    # At any point you can hit Ctrl + C to break out of training early.
    all_start = time.time()
    if is_train:
        try:
            print('begin training')
            for epoch in range(1, params['epochs'] + 1):
                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, params['batch_size'])
                val_rse, val_rmse, val_mape, val_mae, detail = evaluate(Data, Data.valid[0], Data.valid[1], model,
                                                                        evaluateL2,
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
                    test_rse, test_rmse, test_mape, test_mae, detail = evaluate(Data, Data.test[0], Data.test[1], model,
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
    test_rse, test_rmse, test_mape, test_mae, detail = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2,
                                                                evaluateL1,
                                                                params['batch_size'])
    print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae  {:5.4f}".format(test_rse, test_rmse,
                                                                                                test_mape, test_mae))
    return [test_rmse, test_mape, test_mae] + detail


def multi_step_direct(data_type, patient, model, model_path, history_minutes=150, horizon=1, suffix='csv', epochs=300,
                      is_train=True, detailed=True, use_all=False, exclusive=False, use_meal=False, normalize=0,
                      use_insulin=False, validation=None):
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
        'save': f'save/{model_path}/{patient}/{model}.pt',
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
    if use_all:
        print('获取全部数据')
        Data.train = get_all_data(data_type, params, exclusive)[0]

    print(Data.rse)
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
    rmse_loss, mape_loss, mae_loss, detail = multioutput_eval(Data, Data.test[0], Data.test[1], model, evaluateL2,
                                                              evaluateL1,
                                                              params['batch_size'], params, detailed=detailed)

    # print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae  {:5.4f}".format(test_rse, test_rmse, test_mape, test_mae))
    return [rmse_loss, mape_loss, mae_loss] + detail


def recurrent_multistep(data_type, patient, model, model_path, history_minutes=60, horizon=12, suffix='csv', epochs=300,
                        is_train=True, detailed=True, use_all=False, exclusive=False, use_meal=False, normalize=0,
                        use_insulin=False, resampling=False):
    """
    Recurrent multi-step predict using models trained by torch.
    递归预测
    :param model: Trained one-step model.
    :param input: Test inputs.
    :param horizon: max predict step.
    :return:
    """
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

    # Determine the type of diabetes from the data set name
    if data_type in interval_15:
        diabetes_type = 2
    else:
        diabetes_type = 1

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
        'save': f'save/{model_path}/{patient}/{model}.pt',
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
    print(params.save)
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

        print(rmse)
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

    # if data_type == 'ohio_data':
    #     Data = Ohio_utility(params['train_data'], params['test_data'], 0.6, 0.2, params['cuda'], params['horizon'],
    #                         params['window'], params['normalize'])
    # else:
    #     Data = Data_utility(params['data'], 0.6, 0.2, params['cuda'], params['horizon'], params['window'],
    #                         params['normalize'])

    Data = Data_utility(data_type, params['data'], params['train_data'], params['test_data'], 0.6, 0.2, params['cuda'],
                        params['horizon'], params['window'],
                        params['normalize'], output_len=params['output_len'], use_meal=use_meal,
                        use_insulin=use_insulin, resampling=resampling)
    print(Data.rse)
    if use_all:
        print('获取全部数据')
        Data.train = get_all_data(data_type, params, exclusive)[0]

    model = eval(params['model'])(params, Data).cuda()

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

    best_val = 10000000
    optim = Optim(
        model.parameters(), params['optim'], params['lr'], params['clip'],
    )
    early_stopping = EarlyStopping(30, verbose=True, path=params['save'] + '.ckp')
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
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rmse {:5.4f} | valid mape  {:5.4f} | valid mae  {:5.4f} '.format(
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
    rmse_loss, mape_loss, mae_loss, detail = recurssive_evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2,
                                                                 evaluateL1,
                                                                 params['batch_size'], diabetes_type=diabetes_type)
    print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae  {:5.4f}".format(test_rse, test_rmse,
                                                                                                test_mape, test_mae))
    return [rmse_loss, mape_loss, mae_loss] + detail

def recurrent_step_train(data_type, model='GRU', epoch=300, time_interval=5, is_train=True, use_all=False,
                         exclusive=False, history_minutes=60,
                         use_meal=False, normalize=0, use_insulin=False, resampling=False):
    dataset = f'data/{data_type}'
    model_path = f'recurrent_step/{data_type}_epoch_{epoch}_{history_minutes}minutes'
    if use_all:
        model_path += '_all'
    if exclusive:
        model_path += '_exclusive'
    if use_meal:
        model_path += '_meal'
    if normalize != 0:
        model_path += f'_n{normalize}'
    if use_insulin:
        model_path += '_insulin'
    if resampling:
        model_path += '_resampling'
    if data_type == 'ohio_data':
        patient_list = ohio_list
    else:
        patient_list = os.listdir(dataset)

    for patient in patient_list:
        print(patient)
        if type(patient) == int:
            suffix = 'csv'
        else:
            patient, suffix = patient.split('.')

        # model_path = f'recurrent_step/{data_type}_epoch_{epoch}_{history_minutes}minutes/{patient}/{model}'
        save_path = f'result/{model_path}/{patient}/{model}.pkl'
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        print(os.path.dirname(save_path) + model)

        res = recurrent_multistep(data_type, patient, model, model_path=model_path, suffix=suffix, is_train=is_train,
                                  epochs=epoch, history_minutes=history_minutes, use_all=use_all, exclusive=exclusive,
                                  use_meal=use_meal, normalize=normalize, use_insulin=use_insulin,
                                  resampling=resampling)

        # save_path = f'result/recurrent_step/{data_type}_epoch_{epoch}_{history_minutes}minutes/{patient}/{model}.pkl'

        # if not os.path.exists(os.path.dirname(save_path)):
        #     os.makedirs(os.path.dirname(save_path))
        # print(save_path)
        joblib.dump(res, save_path)


def one_step_train(data_type, model, time_step, time_interval, epoch=1000, is_train=False, use_all=False,
                   exclusive=False, history_minutes=60, use_meal=False, normalize=0, use_insulin=False):
    data_type = data_type
    dataset = f'data/{data_type}'
    model_path = f'one_step/{data_type}_epoch_{epoch}_{history_minutes}minutes'
    if use_all:
        model_path += '_all'
    if exclusive:
        model_path += '_exclusive'
    if use_meal:
        model_path += '_meal'
    if normalize != 0:
        model_path += f'_n{normalize}'
    if use_insulin:
        model_path += '_insulin'
    if data_type == 'ohio_data':
        for patient in ohio_list:
            print(patient)
            model = model
            # model_path = f'one_step/{data_type}_epoch_{epoch}/{patient}'
            save_path = f'result/{model_path}/{patient}/{model}.pkl'
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            print(os.path.dirname(save_path) + model)
            # res = {}
            # for i in time_step:
            #     print('==========current horizon============', i)
            #     res[f'{i * time_interval}minuts'] = one_step(data_type, patient, model, i,save_path=model_path)
            # res = pd.DataFrame(res, index=['res', 'rmse', 'mape', 'mae'])
            # if not os.path.exists(os.path.dirname(f'result/one_step/{data_type}_epoch_1000/{patient}/')):
            #     os.makedirs(os.path.dirname(f'result/one_step/{data_type}_epoch_1000/{patient}/'))
            # res.to_csv(f'result/one_step/{data_type}_epoch_1000/{patient}/{model}.csv')
            res = {}
            for i in time_step:
                print('==========current horizon============', i)
                res[f'{i * time_interval}minuts'] = one_step(data_type, patient, model, i,
                                                             is_train=is_train, time_interval=time_interval,
                                                             epochs=epoch, model_path=model_path,
                                                             history_minutes=history_minutes, use_all=use_all,
                                                             exclusive=exclusive, use_meal=use_meal,
                                                             normalize=normalize, use_insulin=use_insulin)

            joblib.dump(res, save_path)

    else:
        for patient in os.listdir(dataset):
            print(patient)
            if type(patient) == int:
                suffix = 'csv'
            else:
                patient, suffix = patient.split('.')

            # model_path = f'one_step/{data_type}_epoch_{epoch}/{patient}'
            save_path = f'result/{model_path}/{patient}/{model}.pkl'
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            print(os.path.dirname(save_path))

            res = {}
            for i in time_step:
                print('==========current horizon============', i)
                res[f'{i * time_interval}minuts'] = one_step(data_type, patient, model, i, suffix,
                                                             is_train=is_train, time_interval=time_interval,
                                                             epochs=epoch, model_path=model_path,
                                                             history_minutes=history_minutes, use_all=use_all,
                                                             exclusive=exclusive, use_meal=use_meal,
                                                             normalize=normalize, use_insulin=use_insulin
                                                             )
            # res=pd.DataFrame(res,index=['res','rmse','mape','mae'])
            # save_path = f'result/one_step/{data_type}_epoch_{epoch}/{patient}/{model}.pkl'
            #
            # if not os.path.exists(os.path.dirname(f'result/one_step/{data_type}_epoch_{epoch}/{patient}/')):
            #     os.makedirs(os.path.dirname(f'result/one_step/{data_type}_epoch_{epoch}/{patient}/'))
            # res.to_csv(f'result/one_step/{data_type}_epoch_1000/{patient}/{model}.csv')
            joblib.dump(res, save_path)


def multi_step_train(data_type, model, epoch=300, is_train=False, use_all=False, exclusive=False, history_minutes=60,
                     use_meal=False, normalize=0, use_insulin=False):
    dataset = f'data/{data_type}'
    model_path = f'multi_step/{data_type}_epoch_{epoch}_{history_minutes}minutes'
    if use_all:
        model_path += '_all'
    if exclusive:
        model_path += '_exclusive'
    if use_meal:
        model_path += '_meal'
    if normalize != 0:
        model_path += f'_n{normalize}'
    if use_insulin:
        model_path += '_insulin'

    if data_type == 'ohio_data':
        patient_list = ohio_list
    else:
        patient_list = os.listdir(dataset)

    for patient in patient_list:
        print(patient)
        print(model_path)
        # if patient!='adolescent#009_simulation_8weeks.pkl':
        #     continue
        if type(patient) == int:
            suffix = 'csv'
        else:
            patient, suffix = patient.split('.')
        save_path = f'result/{model_path}/{patient}/{model}.pkl'

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        print(os.path.dirname(save_path))
        # res = multi_step_direct(data_type, patient, model, epochs=epoch,suffix=suffix)
        res = multi_step_direct(data_type, patient, model, model_path=model_path, is_train=is_train, epochs=epoch,
                                suffix=suffix, history_minutes=history_minutes,
                                use_all=use_all, exclusive=exclusive, use_meal=use_meal,
                                normalize=normalize, use_insulin=use_insulin)
        joblib.dump(res, save_path)


def multi_transfer_train(data_type, model, val_data_type, prediction_type, epoch=300, is_train=False, use_all=False,
                         exclusive=False, history_minutes=60,
                         use_meal=False, normalize=0, use_insulin=False, downsampling=False):
    print(f'{data_type} transfer {val_data_type}')
    dataset = f'data/{data_type}'
    model_path = f'{prediction_type}/{data_type}_epoch_{epoch}_{history_minutes}minutes'
    if use_all:
        model_path += '_all'
    if exclusive:
        model_path += '_exclusive'
    if use_meal:
        model_path += '_meal'
    if normalize != 0:
        model_path += f'_n{normalize}'
    if use_insulin:
        model_path += '_insulin'
    if downsampling:
        model_path += '_downsampling'

    if data_type == 'ohio_data':
        patient_list = ohio_list
    else:
        patient_list = os.listdir(dataset)

    for patient in patient_list:
        print(patient)
        print(model_path)
        # if patient!='adolescent#009_simulation_8weeks.pkl':
        #     continue
        if type(patient) == int:
            suffix = 'csv'
        else:
            patient, suffix = patient.split('.')
        save_path = f'result/{model_path}_transfer/{val_data_type}/{model}.pkl'

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        print(os.path.dirname(save_path))
        # res = multi_step_direct(data_type, patient, model, epochs=epoch,suffix=suffix)
        if prediction_type == 'multi_step':
            res = multi_step_transfer(data_type, patient, model, val_data_type=val_data_type, model_path=model_path,
                                      is_train=is_train, epochs=epoch,
                                      suffix=suffix,
                                      history_minutes=history_minutes,
                                      use_all=use_all, exclusive=exclusive, use_meal=use_meal,
                                      normalize=normalize, use_insulin=use_insulin, )
        elif prediction_type == 'one_step':
            res = {}
            if data_type in interval_15:
                horizons = [2, 4]
                time_interval = 15
            else:
                horizons = [6, 12]
                time_interval = 5
            for i in horizons:
                res[f'{i * time_interval}minuts'] = direct_transfer(data_type, patient, model, horizon=i,
                                                                    val_data_type=val_data_type, model_path=model_path,
                                                                    is_train=is_train, epochs=epoch,
                                                                    suffix=suffix,
                                                                    history_minutes=history_minutes,
                                                                    use_all=use_all, exclusive=exclusive,
                                                                    use_meal=use_meal,
                                                                    normalize=normalize, use_insulin=use_insulin)

        elif prediction_type == 'recurrent_step':
            res = recursive_transfer(data_type, patient, model, val_data_type=val_data_type, model_path=model_path,
                                     is_train=is_train, epochs=epoch,
                                     suffix=suffix,
                                     history_minutes=history_minutes,
                                     use_all=use_all, exclusive=exclusive, use_meal=use_meal,
                                     normalize=normalize, use_insulin=use_insulin, downsampling=downsampling)

        joblib.dump(res, save_path)
        break


if __name__ == '__main__':


    multi_step_train("ShanghaiT1DM", 'GRU', epoch=1000, is_train=True, use_all=True)
    multi_step_train("ShanghaiT2DM", 'GRU', epoch=1000, is_train=True, use_all=True)

    recurrent_step_train("ShanghaiT1DM", 'GRU', epoch=1000, is_train=True, use_all=True)
    recurrent_step_train("ShanghaiT2DM", 'GRU', epoch=1000, is_train=True, use_all=True)

    import Seq2Seq_train as Seq2Seq_train

    Seq2Seq_train.multi_step_train('ShanghaiT2DM', 'Seq2Seq', epoch=1000, is_train=True, history_minutes=60)
    Seq2Seq_train.multi_step_train('ShanghaiT1DM', 'Seq2Seq', epoch=1000, is_train=True, history_minutes=60)

    multi_step_train("ShanghaiT1DM", 'GRU', epoch=1000, is_train=True, history_minutes=60, use_meal=True, normalize=2)
    multi_step_train("ShanghaiT2DM", 'GRU', epoch=1000, is_train=True, history_minutes=60, use_meal=True, normalize=2)

    # downsampling
    # Transfer ShanghaiT1DM trained model to ShanghaiT2DM, downsampling is used for datasets whose time ineterval is 5 minute.
    multi_transfer_train("ShanghaiT1DM", 'GRU', 'ShanghaiT2DM', 'recurrent_step', epoch=1000, is_train=True,
                         history_minutes=60, downsampling=True)
