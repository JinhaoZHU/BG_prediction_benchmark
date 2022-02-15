from util.load_data import *
from util.pytorchtools import *

from models.ClassicalModels import *
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import os

# GAN多步预测训练方法
import argparse
import time
from util.data_utils import *
from util.optim import *
from models.DeepModels import *
from models.Seq2Seq import *
from util.metric import *
from util.pytorchtools import *
from torch.autograd import Variable
import torch
from torch import nn
import os

# 合理的Seq2Seq预测方式

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


def multi_step(data_type, patient, model, model_path, horizon=1, history_minutes=150, suffix='csv', epochs=300,
               is_train=True, detailed=True, use_all=False, exclusive=False, use_meal=False, normalize=0,
                      use_insulin=False):
    prediction_type = 'multi_step'

    class Dict2Obj(dict):
        def __init__(self, *args, **kwargs):
            super(Dict2Obj, self).__init__(*args, **kwargs)

        def __getattr__(self, key):
            value = self[key]
            if isinstance(value, dict):
                value = Dict2Obj(value)
            return value

    if data_type == 'hospital_data':
        time_interval = 15
    else:
        time_interval = 5

    params = {
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
        'hidRNN': 128,
        'hidCNN': 50,
        'hidSkip': 0,
        'L1Loss': False,
        'epochs': epochs,
        'batch_size': 512,
        'output_fun': 'linear',
        'dropout': 0.2,
        # 'save': f'save/{prediction_type}/{data_type}/{patient}/{model}.pt',
        'save': f'save/{model_path}/{model}.pt',
        'clip': 10,
        'seed': 12345,
        'log_interval': 2000,
        'optim': 'adam',
        'lr': 0.001,
        'normalize': 0,
        'gpu': 0,
        'cuda': 1,
        'm': 1
    }

    params = Dict2Obj(params)
    print(os.path.dirname(params.save))
    if not os.path.exists(os.path.dirname(params.save)):
        os.makedirs(os.path.dirname(params.save))

    @torch.no_grad()
    def evaluate(data, X, Y, encoder, decoder, evaluateL2, evaluateL1, batch_size, params):
        encoder.eval()
        decoder.eval()

        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        for X, Y in data.get_batches(X, Y, batch_size, False):
            loss = 0
            input_variable = X
            target_variable = Y

            encoder_hidden = encoder.initHidden(X.size()[0])
            encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

            decoder_input = X[:, -1:, :].transpose(0, 1)
            decoder_input = decoder_input.cuda() if params['cuda'] else decoder_input

            decoder_hidden = encoder_hidden
            decoder_outputs = torch.zeros(Y.size())
            for di in range(Y.size()[1]):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

                decoder_outputs[:, di:di + 1] = decoder_output
                decoder_input = decoder_output.unsqueeze(0)
                loss += evaluateL2(decoder_output, target_variable[:, di:di + 1])
            total_loss += loss.item()
            n_samples += Y.size()[1]

            if predict is None:
                predict = decoder_outputs
                test = Y
            else:
                predict = torch.cat((predict, decoder_outputs))
                test = torch.cat((test, Y))

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

        for i in range(params['output_len']):
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
                                                clarke_loss, parkes_loss, test, predict,Data.test_time]

    def train(data, X, Y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size, params,
              teacher_forcing_ratio=0.5):
        encoder.train();
        decoder.train()

        total_loss = 0
        n_samples = 0
        for X, Y in data.get_batches(X, Y, batch_size, True):
            # X=[batch_size,window,1]
            # Y=[batch_size,output_len]
            input_variable = X
            target_variable = Y

            encoder_hidden = encoder.initHidden(X.size()[0])
            encoder.zero_grad()
            decoder.zero_grad()

            input_length = X.size()[1]
            target_length = Y.size()[1]

            encoder_outputs = Variable(torch.zeros(params['window'], encoder.hidden_size))
            encoder_outputs = encoder_outputs.cuda() if params['cuda'] else encoder_outputs

            loss = 0
            # encoder_output=[seq_len,batch_size,hidden_size]
            # encoder_hidden=[1,batch_size,hidden_size]

            encoder_outputs, encoder_hidden = encoder(X, encoder_hidden)

            decoder_input = X[:, -1:, :].transpose(0, 1)
            decoder_input = decoder_input.cuda() if params['cuda'] else decoder_input
            decoder_hidden = encoder_hidden

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                # 利用已知的上一步真实的单词去预测下一个单词
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)

                    loss += criterion(decoder_output, target_variable[:, di:di + 1])
                    decoder_input = target_variable[:, di:di + 1].unsqueeze(0)
            else:
                # 利用自己上一步预测的单词作为输入预测下一个单词
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)

                    decoder_input = decoder_output.unsqueeze(0)

                    loss += criterion(decoder_output, target_variable[:, di:di + 1])

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            total_loss += loss.item()
            n_samples += target_length
        return total_loss / n_samples

    # if data_type == 'ohio_data':
    #     Data = Ohio_utility(params['train_data'], params['test_data'], 0.6, 0.2, params['cuda'], params['horizon'],
    #                         params['window'], params['normalize'], output_len=params['output_len'])
    # else:
    #     Data = Data_utility(params['data'], 0.6, 0.2, params['cuda'], params['horizon'], params['window'],
    #                         params['normalize'], output_len=params['output_len'])

    Data = Data_utility(data_type, params['data'], params['train_data'], params['test_data'], 0.6, 0.2, params['cuda'],
                        params['horizon'], params['window'],
                        params['normalize'], output_len=params['output_len'], use_meal=use_meal,
                        use_insulin=use_insulin)
    if use_all:
        print('获取全部数据')
        Data.train = get_all_data(data_type, params, exclusive)

    print(Data.rse)
    # print(Data.train[0].shape)
    # print(Data.train[1].shape)
    # model = eval(params['model'])(params, Data)
    if model == 'Seq2Seq':
        encoder = EncoderRNN(Data.m, 256)
        decoder = DecoderRNN(256, Data.m)
    elif model == 'Seq2Seq_Attention':
        encoder = EncoderRNN(Data.m, 256)
        decoder = AttnDecoderRNN(256, Data.m,max_length=params['window'])

    nParams = sum([p.nelement() for p in encoder.parameters()])
    print('* number of parameters: %d' % nParams)

    if params['L1Loss']:
        criterion = nn.L1Loss(size_average=False)
    else:
        criterion = nn.MSELoss(size_average=False)

    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()

    best_val = 10000000
    encoder_optimizer = Optim(encoder.parameters(), params['optim'], params['lr'], params['clip'])
    decoder_optimizer = Optim(decoder.parameters(), params['optim'], params['lr'], params['clip'])

    early_stopping = EarlyStopping(30, verbose=True, path=params['save'] + '.ckp',model_type=model)
    # At any point you can hit Ctrl + C to break out of training early.
    if not os.path.exists(os.path.dirname(params['save'])):
        os.makedirs(os.path.dirname(params['save']))

    all_start = time.time()
    if is_train:
        try:
            print('begin training')
            for epoch in range(1, params['epochs'] + 1):
                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], encoder, decoder,
                                   encoder_optimizer, decoder_optimizer, criterion,
                                   params['batch_size'], params)

                val_rmse, val_mape, val_mae, other = evaluate(Data, Data.valid[0], Data.valid[1], encoder, decoder,
                                                              evaluateL2,
                                                              evaluateL1,
                                                              params['batch_size'], params)
                val_rmse = np.mean(val_rmse)
                val_mape = np.mean(val_mape)
                val_mae = np.mean(val_mae)
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rsme {:5.4f} | valid mape  {:5.4f} | valid mae  {:5.4f} '.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_rmse, val_mape, val_mae))
                # Save the model if the validation loss is the best we've seen so far.

                if val_rmse < best_val:
                    with open(params['save'], 'wb') as f:
                        torch.save([encoder, decoder], f)
                    best_val = val_rmse
                if epoch % 5 == 0:
                    test_rmse, test_mape, test_mae, other = evaluate(Data, Data.test[0], Data.test[1], encoder,
                                                                     decoder, evaluateL2, evaluateL1,
                                                                     params['batch_size'], params)
                    test_rmse = np.mean(test_rmse)
                    test_mape = np.mean(test_mape)
                    test_mae = np.mean(test_mae)
                    print(
                        "test rmse {:5.4f} | test mape {:5.4f} | test mae {:5.4f}".format(test_rmse,
                                                                                                             test_mape,
                                                                                                             test_mae))
                early_stopping(val_rmse, [encoder, decoder])
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
        [encoder, decoder] = torch.load(f)
    rmse_loss, mape_loss, mae_loss, detail = evaluate(Data, Data.test[0], Data.test[1], encoder,
                                                      decoder, evaluateL2, evaluateL1,
                                                      params['batch_size'], params)
    # print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae  {:5.4f}".format(test_rse, test_rmse,
    #                                                                                             test_mape, test_mae))
    return [rmse_loss, mape_loss, mae_loss] + detail


import os
import pandas as pd


# data_type='hospital_data'
# dataset=f'data/{data_type}'
# for patient in os.listdir(dataset):
#   print(patient)
#   patient=patient.split('.')[0]
#   model='Seq2Seq'
#   res=multi_step(data_type,patient,model)
#   res=pd.DataFrame(res,index=['rmse','mape','mae'])
#   res.to_csv(f'result/multi_step/{data_type}/{patient}/{model}.csv')

def multi_step_train(data_type, model, epoch=300, is_train=False, use_all=False, exclusive=False, history_minutes=60):
    dataset = f'data/{data_type}'
    model_path = f'multi_step/{data_type}_epoch_{epoch}_{history_minutes}minutes'

    if use_all:
        model_path += '_all'
    if exclusive:
        model_path += '_exclusive'
    if data_type == 'ohio_data':
        for patient in [540, 544, 552, 567, 584, 596,559,563,570,575,588,591]:
            print(patient,model)
            save_path = f'result/{model_path}/{patient}/{model}.pkl'
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            print(os.path.dirname(save_path))

            res = multi_step(data_type, patient, model, suffix='pkl',
                             is_train=is_train, epochs=epoch, model_path=model_path,
                             history_minutes=history_minutes, use_all=use_all, exclusive=exclusive)
            # res = pd.DataFrame(res, index=['rmse', 'mape', 'mae'])
            # if not os.path.exists(f'{save_path}/{patient}/'):
            #     os.makedirs(os.path.dirname(f'{save_path}/{patient}/'))
            # res.to_csv(f'{save_path}/{patient}/{model}.csv')
            joblib.dump(res, save_path)
    else:
        for patient in os.listdir(dataset):
            print(patient,model)
            patient, suffix = patient.split('.')
            save_path = f'result/{model_path}/{patient}/{model}.pkl'
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            print(os.path.dirname(save_path))
            res = multi_step(data_type, patient, model, is_train=is_train, epochs=epoch, suffix=suffix,
                             history_minutes=history_minutes,
                             model_path=model_path, use_all=use_all, exclusive=exclusive
                             )
            # res = pd.DataFrame(res, index=['rmse', 'mape', 'mae'])
            # if not os.path.exists(f'{save_path}/{patient}/'):
            #     os.makedirs(os.path.dirname(f'{save_path}/{patient}/'))
            # res.to_csv(f'{save_path}/{patient}/{model}.csv')
            joblib.dump(res, save_path)


if __name__ == '__main__':
    # multi_step_train('hospital_data', 'Seq2Seq', epoch=1000, is_train=True, history_minutes=60)
    # multi_step_train('ohio_data', 'Seq2Seq', epoch=1000, is_train=True, history_minutes=60)
    # multi_step_train('simulator_data', 'Seq2Seq', epoch=1000, is_train=True, history_minutes=60)
    #
    # multi_step_train('hospital_data', 'Seq2Seq', epoch=1000, is_train=True, history_minutes=240)
    # multi_step_train('ohio_data', 'Seq2Seq', epoch=1000, is_train=True, history_minutes=240)
    # multi_step_train('simulator_data', 'Seq2Seq', epoch=1000, is_train=True, history_minutes=240)
    #
    # multi_step_train('hospital_data', 'Seq2Seq_Attention', epoch=1000, is_train=True, history_minutes=60)
    # multi_step_train('ohio_data', 'Seq2Seq_Attention', epoch=1000, is_train=True, history_minutes=60)
    # multi_step_train('simulator_data', 'Seq2Seq_Attention', epoch=1000, is_train=True, history_minutes=60)
    #
    # multi_step_train('hospital_data', 'Seq2Seq_Attention', epoch=1000, is_train=True, history_minutes=240)
    # multi_step_train('ohio_data', 'Seq2Seq_Attention', epoch=1000, is_train=True, history_minutes=240)
    multi_step_train('simulator_data', 'Seq2Seq_Attention', epoch=1000, is_train=True, history_minutes=240)
