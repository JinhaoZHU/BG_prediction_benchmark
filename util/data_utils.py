import torch
import numpy as np;
from torch.autograd import Variable
import pandas as pd  # a
import joblib
import os

np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

ohio_list = [540, 544, 552, 567, 584, 596, 559, 563, 570, 575, 588, 591]
interval_15 = ['hospital_data', 'EastT1DM']


def normal_std(x):
    # 求总体标准差
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


def ape(pred, Y):
    return torch.mean(torch.abs((pred - Y) / Y))


def get_time(x):
    dic = {}
    # dic['hour']=int(x.name.hour)
    # dic['minute']=int(x.name.minute)
    dic['hour'] = str(x.name.hour // 3)
    dic['minute'] = str(x.name.minute // 15)
    return pd.Series(dic)


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, data_type, file_name, train_data, test_data, train, valid, cuda, horizon, window, normalize=0,
                 output_len=1, time_encoding=False, use_meal=False, use_insulin=False,resampling=False,downsampling=False):
        self.cuda = cuda;
        self.P = window;
        self.h = horizon
        if output_len != 1:
            self.h = output_len
        resampling=resampling or downsampling
        # 是否使用实际值
        self.real_values = True
        self.data_type = data_type
        if data_type in ['hospital_data', 'EastT1DM']:
            data = pd.read_csv(file_name, index_col=0).dropna(axis=0, subset=["cgm"])
            fin = data.iloc[:, :1]
            fin = fin.dropna(axis=0)
            fin.index = pd.to_datetime(fin.index)
            if time_encoding:
                fin[['hour', 'minute']] = fin.apply(get_time, axis=1)
                fin = fin[['cgm', 'hour', 'minute']]
                fin = pd.get_dummies(fin[['cgm', 'hour', 'minute']], columns=['hour', 'minute'])
            if use_meal:
                fin.loc[:, 'meal'] = data['meal']
        elif data_type == 'simulator_data':
            data = joblib.load(file_name)[0][0]
            if resampling:
                data=data.resample('15min').first()
            data[['hour', 'minute']] = data.apply(get_time, axis=1)
            fin = data[['CGM']].copy()
            if time_encoding:
                fin = pd.concat(fin, pd.get_dummies(data[['hour', 'minute']], columns=['hour', 'minute']), axis=1)
            if use_meal:
                fin.loc[:, 'meal'] = data['CHO']
            if use_insulin:
                fin.loc[:, 'insulin'] = data['insulin']
        elif data_type == 'ohio_data':
            data_train = pd.read_csv(train_data, index_col=0)
            data_train.index = pd.to_datetime(data_train.index)
            data_train['gl_value'] = data_train['gl_value'].interpolate()
            data_train = data_train.dropna(axis=0, subset=["gl_value"])
            train_len = data_train.shape[0]

            data_test = pd.read_csv(test_data, index_col=0)
            data_test.index = pd.to_datetime(data_test.index)
            data_test['gl_value'] = data_test['gl_value'].interpolate(method='pad')
            test_len = data_test.shape[0]
            merged = pd.concat([data_train, data_test])
            if resampling:
                merged=merged.resample('15min').first()
            fin = merged[['gl_value']].copy()
            if use_meal:
                fin.loc[:, 'meal'] = merged['meal_carbs'].fillna(0)
            if time_encoding:
                fin.index = pd.to_datetime(fin.index)
                fin[['hour', 'minute']] = fin.apply(get_time, axis=1)
                fin = fin[['gl_value', 'hour', 'minute']]
                fin = pd.get_dummies(fin[['gl_value', 'hour', 'minute']], columns=['hour', 'minute'])
        self.rawdat = fin.values
        self.dat = np.zeros(self.rawdat.shape);
        self.n, self.m = self.dat.shape;  # n为数据条数，m为数据维度
        self.normalize = normalize
        self.scale = np.ones(self.m);
        self._normalized(normalize);
        # 划分数据集,x为所有输入变量，y为血糖值
        self._split(int(train * self.n), int((train + valid) * self.n), self.n);

        self.test_time = fin.iloc[int((train + valid) * self.n) - output_len + 1:self.n, :]

        self.scale = torch.from_numpy(self.scale).float();
        # 将Y的值放到一个[1,m]大小的float类型的tensor上

        tmp = self.test[1] * self.scale[0].expand(self.test[1].size(0), self.test[1].size(1));

        if self.cuda:
            self.scale = self.scale.cuda();
        self.scale = Variable(self.scale);

        self.rse = normal_std(tmp);  # 求总体标准差，均方误差
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)));  # 求平均绝对误差

        self.output_dim = 361
        self.bin_step = (400 - 40) / (self.output_dim - 1)
        # the half step appraoch is an artifact of wanting perfect bins with output_dim=361
        self.bins = self.bins = np.linspace(40, 400, self.output_dim) + (self.bin_step * 0.5)

    def values_to_bins(self, y):
        '''
        将具体的血糖值转化为40-400的区间的位置，比如40就代表0号位
        :param y: 形状为[batch_size,seq_len,1]
        :return: [batch_size,seq_len,361]
        '''
        if self.real_values:
            return y
        else:
            return np.digitize(y, self.bins)

    def bins_to_values(self, y):
        '''
        将类别结果转换成实际的血糖值
        '''
        if type(y) is not np.ndarray:
            y = y.numpy()
        if self.real_values:
            return y
        else:
            vals = self.bins[np.clip(np.array(y, dtype=int), 0, self.output_dim - 1)]
            return vals - (0.5 * self.bin_step)

    def one_hot(self, seq):
        '''
        turn glucose signal into one hot distribution
        with size=output_dim, linearly bins glucose
        range 40-400
        don't need for NLLLoss
        '''
        dist = np.zeros((seq.size, self.output_dim))
        dist[np.arange(seq.size), np.digitize(seq, self.bins)] = 1.
        return dist

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat);

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]));
                if self.scale[i] == 0:
                    self.dat[:, i] = self.rawdat[:, i]
                else:
                    self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]));

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train);
        valid_set = range(train, valid);
        test_set = range(valid, self.n);
        # if self.data_type=='ohio_data':
        #     test_set = range(valid + self.h + 1 + self.P, self.n);
        self.train = self._batchify(train_set, self.h);
        self.valid = self._batchify(valid_set, self.h);
        self.test = self._batchify(test_set, self.h);

    def _batchify(self, idx_set, horizon):

        n = len(idx_set);
        X = torch.zeros((n, self.P, self.m));
        Y = torch.zeros((n, self.h));

        for i in range(n):
            end = idx_set[i] - self.h + 1;
            start = end - self.P;
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :]);  # X.shape=(size,P,m)size为数据集大小，P为窗口大小,m为数据维度
            Y[i, :] = torch.from_numpy(self.dat[end:idx_set[i] + 1, :1]).reshape(1,
                                                                                 -1);  # Y.shape=(size,m) 通过start-end这P个数据预测h时间后的数据
        return [X, Y];

    # 分batch获取数据
    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt];
            Y = targets[excerpt];
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();
            yield Variable(X), Variable(Y);
            start_idx += batch_size


class Ohio_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, train_data, test_data, train, valid, cuda, horizon, window, normalize=2, output_len=1,
                 time_encoding=False, use_meal=False, use_insulin=False):
        self.cuda = cuda;
        self.P = window;
        self.h = horizon
        if output_len != 1:
            self.h = output_len
        # 是否使用实际值
        self.real_values = True
        # fin = open(file_name);
        # self.rawdat = np.loadtxt(fin,delimiter=',')
        # if file_name.split('.')[-1] == 'csv':
        #     fin = pd.read_csv(file_name, index_col=0)
        # elif file_name.split('.')[-1] == 'pkl':
        #     fin = joblib.load(file_name)[0][0][['CGM']]

        data_train = pd.read_csv(train_data, index_col=0)
        data_train.index = pd.to_datetime(data_train.index)
        data_train['gl_value'] = data_train['gl_value'].interpolate()
        data_train = data_train.dropna(axis=0, subset=["gl_value"])
        train_len = data_train.shape[0]

        data_test = pd.read_csv(test_data, index_col=0)
        data_test.index = pd.to_datetime(data_test.index)
        data_test['gl_value'] = data_test['gl_value'].interpolate(method='pad')
        test_len = data_test.shape[0]
        merged = pd.concat([data_train, data_test])

        fin = merged[['gl_value']].copy()
        if use_meal:
            fin.loc[:, 'meal'] = merged['meal_carbs'].fillna(0)
        if time_encoding:
            fin.index = pd.to_datetime(fin.index)
            fin[['hour', 'minute']] = fin.apply(get_time, axis=1)
            fin = fin[['gl_value', 'hour', 'minute']]
            fin = pd.get_dummies(fin[['gl_value', 'hour', 'minute']], columns=['hour', 'minute'])

        all_len = train_len + test_len
        self.rawdat = fin.values
        self.dat = np.zeros(self.rawdat.shape);
        self.n, self.m = self.dat.shape;  # n为数据条数，m为数据维度
        self.normalize = 2
        self.scale = np.ones(self.m);
        self._normalized(normalize);

        self._split(int(train_len * 0.8), int(train_len), self.n);

        self.scale = torch.from_numpy(self.scale).float();
        # 将Y的值放到一个[1,m]大小的float类型的tensor上
        # tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m);
        tmp = self.test[1] * self.scale[0].expand(self.test[1].size(0), self.test[1].size(1));

        if self.cuda:
            self.scale = self.scale.cuda();
        self.scale = Variable(self.scale);

        self.rse = normal_std(tmp);  # 求总体标准差，均方误差
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)));  # 求平均绝对误差

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat);

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]));
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]));

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train);
        valid_set = range(train, valid);
        test_set = range(valid + self.h + 1 + self.P, self.n);
        self.train = self._batchify(train_set, self.h);
        self.valid = self._batchify(valid_set, self.h);
        self.test = self._batchify(test_set, self.h);

    def _batchify(self, idx_set, horizon):

        n = len(idx_set);
        X = torch.zeros((n, self.P, self.m));
        Y = torch.zeros((n, self.h));

        for i in range(n):
            end = idx_set[i] - self.h + 1;
            start = end - self.P;
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :]);  # X.shape=(size,P,m)size为数据集大小，P为窗口大小,m为数据维度
            Y[i, :] = torch.from_numpy(self.dat[end:idx_set[i] + 1, :1]).reshape(1,
                                                                                 -1);  # Y.shape=(size,m) 通过start-end这P个数据预测h时间后的数据
        return [X, Y];

    # 分batch获取数据
    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt];
            Y = targets[excerpt];
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();
            yield Variable(X), Variable(Y);
            start_idx += batch_size


def get_all_data(data_type, params, exclusive,resampling=False,downsampling=False):
    '''
    获取同一数据集中的所有数据作为训练集
    :param data_type:
    :param params:
    :return:
    '''
    data_set = f'data/{data_type}'
    train_data = None
    val_data = None
    test_data = None
    save_path = f'data/all_data/{data_type}'
    resampling=resampling or downsampling
    if exclusive:
        save_path += '_exclusive_'
        save_path += str(params['patient'])
    if resampling:
        save_path += '_resampling'
    save_path += '.pkl'
    if os.path.exists(save_path):
        return joblib.load(save_path)
    if data_type == 'ohio_data':
        for patient in ohio_list:
            # print(patient)
            train_path = f'data/{data_type}/{patient}-ws-training.csv'
            test_path = f'data/{data_type}/{patient}-ws-testing.csv'
            if train_path == params['train_data'] and exclusive:
                continue
            Data = Data_utility(data_type, params['data'], train_path, test_path, 0.6, 0.2, params['cuda'],
                                params['horizon'], params['window'],
                                params['normalize'], output_len=params['output_len'],resampling=resampling)
            if train_data is None:
                train_data = []
                val_data = []
                test_data = []
                train_data.append(Data.train[0])
                train_data.append(Data.train[1])
                val_data.append(Data.valid[0])
                val_data.append(Data.valid[1])
                test_data.append(Data.test[0])
                test_data.append(Data.test[1])
            else:
                train_data[0] = torch.cat((train_data[0], Data.train[0]))
                train_data[1] = torch.cat((train_data[1], Data.train[1]))
                val_data[0] = torch.cat((val_data[0], Data.valid[0]))
                val_data[1] = torch.cat((val_data[1], Data.valid[1]))
                test_data[0] = torch.cat((test_data[0], Data.test[0]))
                test_data[1] = torch.cat((test_data[1], Data.test[1]))
    else:
        for patient in os.listdir(data_set):
            # print(patient)
            patient, suffix = patient.split('.')
            if f'data/{data_type}/{patient}.{suffix}' == params['data'] and exclusive:
                continue
            # Data = Data_utility(f'data/{data_type}/{patient}.{suffix}', 0.6, 0.2, params['cuda'], params['horizon'],
            #                     params['window'],
            #                     params['normalize'], output_len=params['output_len'])

            Data = Data_utility(data_type, f'data/{data_type}/{patient}.{suffix}', params['train_data'],
                                params['test_data'], 0.6, 0.2,
                                params['cuda'],
                                params['horizon'], params['window'],
                                params['normalize'], output_len=params['output_len'],resampling=resampling)

            if train_data is None:
                train_data = []
                val_data = []
                test_data = []
                train_data.append(Data.train[0])
                train_data.append(Data.train[1])
                val_data.append(Data.valid[0])
                val_data.append(Data.valid[1])
                test_data.append(Data.test[0])
                test_data.append(Data.test[1])
            else:
                train_data[0] = torch.cat((train_data[0], Data.train[0]))
                train_data[1] = torch.cat((train_data[1], Data.train[1]))
                val_data[0] = torch.cat((val_data[0], Data.valid[0]))
                val_data[1] = torch.cat((val_data[1], Data.valid[1]))
                test_data[0] = torch.cat((test_data[0], Data.test[0]))
                test_data[1] = torch.cat((test_data[1], Data.test[1]))
    joblib.dump([train_data, val_data, test_data], save_path)
    return [train_data, val_data, test_data]
