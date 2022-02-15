"""
Deep Models:
    * NN (Naive)
    * MLP
    * CNN
    * RNN
    * RNN (LSTM)
    * RNN (GRU)
    * LSTNet (ref: https://github.com/laiguokun/LSTNet)
Implemented by PyTorch.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import random


class ForecastRNN(nn.Module):
    def __init__(self):
        super(ForecastRNN,self).__init__()

    @staticmethod
    def _dist_to_bins(dist):
        return torch.max(dist, dim=-1)[1]

    @staticmethod
    def _get_sequence_info(seq):
        batch_size = seq.size(0)
        sequence_length = seq.size(1)
        return batch_size, sequence_length

class NN(ForecastRNN):
    def __init__(self, args, data):
        super(NN, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear = nn.Linear(self.P * self.m, self.m)
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        c = x.view(-1, self.P * self.m)
        return self.output(self.linear(c))


class MLP(nn.Module):
    def __init__(self, args, data):
        super(MLP, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.dropout = nn.Dropout(p=args.dropout)

        self.n1 = int(self.P * self.m * 1.5)
        self.n2 = self.n1

        self.dense1 = nn.Linear(self.P * self.m, self.n1)
        self.dense2 = nn.Linear(self.n1, self.n2)
        self.dense3 = nn.Linear(self.n2, self.m)

        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        x = x.view(-1, self.P * self.m)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.output(self.dense3(x))
        if self.output:
            x = self.output(x)
        return self.dropout(x)


class RNN(nn.Module):
    def __init__(self, args, data):
        super(RNN, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.dropout = nn.Dropout(p=args.dropout)
        self.hidR = args.hidRNN
        self.RNN = nn.RNN(
            input_size=self.m,
            hidden_size=self.hidR,
            batch_first=True
        )
        self.linear = nn.Linear(self.hidR, self.m)
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        _, h = self.RNN(x)
        h = h.contiguous().view(-1, self.hidR)
        h = self.dropout(h)
        o = self.linear(h)
        if self.output:
            o = self.output(o)
        return o


class GRU(ForecastRNN):
    def __init__(self, args, data):
        super(GRU, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.horizon = args.horizon
        self.output_len = args.output_len
        self.dropout = nn.Dropout(p=args.dropout)
        self.hidR = args.hidRNN
    
        self.RNN = nn.GRU(
            input_size=self.m,
            hidden_size=self.hidR,
            batch_first=True
        )
        self.linear = nn.Linear(self.hidR, self.output_len)
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        _, h = self.RNN(x)
        h = h.view(-1, self.hidR)
        h = self.dropout(h)
        o = self.linear(h)
        if self.output:
            o = self.output(o)
        return o


class GRU_Time(nn.Module):
    def __init__(self, args, data):
        super(GRU_Time, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = 1
        self.horizon = args.horizon
        self.output_len = args.output_len
        self.dropout = nn.Dropout(p=args.dropout)
        self.hidR = args.hidRNN
        self.RNN = nn.GRU(
            input_size=self.m,
            hidden_size=self.hidR,
            batch_first=True
        )
        self.linear = nn.Linear(self.hidR + 2, self.output_len)
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        time_info = x[:, -1, -2:]
        x = x[:, :, :1]
        _, h = self.RNN(x)
        h = h.view(-1, self.hidR)
        h = self.dropout(h)
        o = self.linear(torch.cat((h, time_info), dim=1))
        if self.output:
            o = self.output(o)
        return o


class LSTM(nn.Module):
    def __init__(self, args, data):
        super(LSTM, self).__init__()
        self.P = args.window
        self.m = data.m
        self.dropout = nn.Dropout(p=args.dropout)
        self.hidR = args.hidRNN
        self.lstm = nn.LSTM(
            input_size=self.m,
            hidden_size=self.hidR,
            batch_first=True
        )
        self.linear = nn.Linear(self.P * self.hidR, self.m)
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        # x     [batch_size, time_step, input_size]
        # r_out [batch_size, time_step, output_size]
        # h_n   [n_layers, batch_size, hidRNN]
        # h_c   [n_layers, batch_size, hidRNN]
        r_out, (h_n, h_c) = self.lstm(x)
        r_out = r_out.contiguous().view(-1, self.P * self.hidR)
        r_out = self.dropout(r_out)
        o = self.linear(r_out)
        if self.output:
            o = self.output(o)
        return o


class CNN(nn.Module):
    def __init__(self, args, data):
        super(CNN, self).__init__()
        self.P = args.window
        self.m = data.m
        self.dropout = nn.Dropout(p=args.dropout)
        self.hidC = args.hidCNN
        self.Ck = args.CNN_kernel
        self.width = self.m
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.width))
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear = nn.Linear(self.hidC * (self.P - self.Ck + 1) * (self.m - self.width + 1), self.m)
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        # x: [batch_size, window, N(=m)]
        # c: [batch_size, window, 1, N]
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        # [batch_size, hid_CNN, window-height+1, N-width+1] CNN_kernel:(height, width)
        c = self.dropout(c)
        c = c.view(-1, self.hidC * (self.P - self.Ck + 1) * (self.m - self.width + 1))
        # output: [batch_size, N]

        c = self.linear(c)
        if self.output:
            c = self.output(c)
        return c


"""
    Ref: https://github.com/laiguokun/LSTNet
         https://arxiv.org/abs/1703.07015
    Implemented by PyTorch.
"""


class LSTNet(nn.Module):
    def __init__(self, args, data):
        super(LSTNet, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = (self.P - self.Ck) // self.skip
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn
        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z

        if self.output:
            res = self.output(res)
        return res


class SeqMo(ForecastRNN):
    def __init__(self, args, data):
        super(SeqMo, self).__init__()
        self.use_cuda = args.cuda  # 是否使用cuda
        self.P = args.window;  # 窗口大小
        self.m = data.m  # 数据的维度
        self.input_dim = data.m
        self.hidR = args.hidRNN;  # number of RNN hidden units，100
        self.output_len = args.output_len
        self.output_dim = 1
        self.dropout = nn.Dropout(p=args.dropout);
        # 第一步对所以的序列提取特征
        self.rnn = nn.GRU(input_size=self.input_dim,
                          hidden_size=self.hidR,
                          num_layers=1,
                          dropout=0,
                          bidirectional=False);  # hidC为输入的单个序列的维度
        # 解码器
        self.decoder = nn.GRU(input_size=self.hidR,
                              hidden_size=self.hidR,
                              num_layers=1,
                              bidirectional=False,
                              dropout=0)

        self.output = nn.Linear(self.hidR, self.output_dim)

        if self.use_cuda:
            self.rnn.cuda()
            self.decoder.cuda()
            self.output.cuda()

    def forward(self, x):
        # x:[batch_size,seq_len,dim]
        batch_size = x.size(0)
        seq_len = x.size(1)
        h = Variable(torch.zeros(self.rnn.num_layers,
                                 batch_size,  # not sure if need to reshape for batch_first
                                 self.rnn.hidden_size).type(torch.FloatTensor),
                     requires_grad=False).cuda()
        x = x.permute(1, 0, 2).contiguous();
        # out=[seq_len,batch_size,num_directions*hidden_size)
        # h=[num_layers*num_direction,batch_size,hidden_size)
        out, h = self.rnn.forward(x, h)

        # out_flat=out.contiguous().view(-1,self.rnn.hidden_size)

        y = []
        # encoded=out_flat[None,:]
        encoded = h
        hidden = Variable(torch.zeros(encoded.data.shape)).cuda()
        for i in range(self.output_len):
            encoded, hidden = self.decoder(encoded, hidden)

            pred = self.output(encoded[0]).contiguous()

            y.append(pred.view(batch_size, self.output_dim))

        y = torch.cat(y, dim=1)
        # final_y=y[:,-1].contiguous().view(batch_size,self.output_len,self.output_dim)
        # return final_y.squeeze(dim=2);
        return y


class Seq2Seq(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.use_cuda = args.cuda  # 是否使用cuda
        self.P = args.window;  # 窗口大小,7*24
        self.m = data.m  # 数据的维度,500
        self.input_dim = data.m
        self.hidR = args.hidRNN;  # number of RNN hidden units，100
        self.output_len = args.horizon
        self.output_dim = 1
        self.dropout = nn.Dropout(p=args.dropout);
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(input_dim=self.input_dim,
                               enc_hid_dim=128,
                               dec_hid_dim=128,
                               dropout=0)

        self.decoder = Decoder(output_dim=self.output_dim,
                               emb_dim=1,
                               enc_hid_dim=128,
                               dec_hid_dim=128,
                               dropout=0,
                               attention=None
                               )
        self.seq2seq = seq2seq_main(self.encoder, self.decoder, device)

        if self.use_cuda:
            self.seq2seq.cuda()

    def forward(self, x, trg, teacher_forcing_ratio=0.5):
        '''

        :param x: [batch_size,seq_len,m]
        :param trg: [batch_size, trg_len]
        :param teacher_forcing_ratio:
        :return:
        '''
        batch_size = x.size(0)
        seq_len = x.size(1)
        final_y = self.seq2seq(x, trg, teacher_forcing_ratio)
        return final_y;


class Seq2Seq_Attention(nn.Module):
    def __init__(self, args, data):
        super(Seq2Seq_Attention, self).__init__()
        self.use_cuda = args.cuda  # 是否使用cuda
        self.P = args.window;  # 窗口大小,7*24
        self.m = data.m  # 数据的维度,500
        self.input_dim = data.m
        self.hidR = args.hidRNN;  # number of RNN hidden units，100
        self.output_len = args.horizon
        self.output_dim = 1
        self.dropout = nn.Dropout(p=args.dropout);
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(input_dim=self.input_dim,
                               enc_hid_dim=128,
                               dec_hid_dim=128,
                               dropout=0)
        self.attention = Attention(enc_hid_dim=128,
                                   dec_hid_dim=128)
        self.decoder = Decoder(output_dim=self.output_dim,
                               emb_dim=257,
                               enc_hid_dim=128,
                               dec_hid_dim=128,
                               dropout=0,
                               attention=self.attention
                               )
        self.seq2seq = seq2seq_main(self.encoder, self.decoder, device)

        if self.use_cuda:
            self.seq2seq.cuda()

    def forward(self, x, trg, teacher_forcing_ratio=0.5):
        '''

        :param x: [batch_size,seq_len,m]
        :param trg: [batch_size, trg_len]
        :param teacher_forcing_ratio:
        :return:
        '''
        batch_size = x.size(0)
        seq_len = x.size(1)
        final_y = self.seq2seq(x, trg, teacher_forcing_ratio)
        return final_y;


class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, dropout, attention=False):
        super().__init__()

        self.rnn = nn.GRU(input_size=input_dim, hidden_size=enc_hid_dim,
                          bidirectional=True,
                          num_layers=1,
                          dropout=0)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, src):
        '''
        :param src: src=[batch_size,seq_len,m]
        :return:
        '''
        # src=src.transpose(0,1)      #src=[batch_size,seq_len]
        # embedded = self.dropout(self.embedding(src)).transpose(0, 1) # embedded = [src_len, batch_size, emb_dim]

        # src=[src_len,batch_size,m]
        src = src.permute(1, 0, 2).contiguous();

        # enc_output = [src_len, batch_size, hid_dim * num_directions]
        # enc_hidden = [n_layers * num_directions, batch_size, hid_dim]
        enc_output, enc_hidden = self.rnn(src)  # if h_0 is not give, it will be set 0 acquiescently

        # enc_hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # enc_output are always from the last layer

        # enc_hidden [-2, :, : ] is the last of the forwards RNN
        # enc_hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hid_dim]
        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1)))

        return enc_output, s


class Attention(nn.Module):
    '''
    Attention层
    主要步骤为：
    1. 计算source和target之间的相似度：有多重方法
    2. 根据相似度计算权重
    3. 根据权重加权求和encoder的输出
    '''

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention=None):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear(dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, s, enc_output):
        # dec_input = [batch_size,1],Decoder的起始输入的序号，文本翻译中为<sos>
        # s = [batch_size, dec_hid_dim],输入到Encoder的前一层隐藏层状态
        # enc_output = [src_len, batch_size, enc_hid_dim * 2],Encoder输出的顶层的隐藏层状态，表示每个输入序列点的隐藏状态

        # 将起始输入转换成嵌入向量
        # dec_input = dec_input.unsqueeze(1) # dec_input = [batch_size, 1]
        # embedded = self.dropout(self.embedding(dec_input)).transpose(0, 1) # embedded = [1, batch_size, emb_dim]
        # dec_input = [1, batch_size, 1]
        # print('dec_input:',dec_input.shape)
        dec_input = self.dropout(dec_input.unsqueeze(2)).transpose(0, 1)

        # 计算前一层隐藏层状态和所有输入序列的隐藏层状态的attention分数，得到的是权重
        # a = [batch_size, 1, src_len]
        if self.attention is not None:
            a = self.attention(s, enc_output).unsqueeze(1)

            # enc_output = [batch_size, src_len, enc_hid_dim * 2]
            enc_output = enc_output.transpose(0, 1)

            # 根据权重，对输入序列的隐藏状态进行加权求和。
            # c = [1, batch_size, enc_hid_dim * 2]
            c = torch.bmm(a, enc_output).transpose(0, 1)

            # 将得到的输入序列的attention结果和上一步输出的隐藏层状态合并
            # rnn_input = [1, batch_size, (enc_hid_dim * 2) + 1]
            rnn_input = torch.cat((dec_input, c), dim=2)
            c = c.squeeze(0)

        else:
            rnn_input = dec_input

        # 将结果输入到rnn网络中，得到输出
        # dec_output = [src_len(=1), batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions=1, batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))

        # 整理输入序列值的嵌入向量，attention结果和rnn输出的结果的形状
        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        embedded = dec_input.squeeze(0)
        dec_output = dec_output.squeeze(0)

        # print(embedded.shape)
        # print(torch.cat((dec_output, embedded), dim=1).shape)
        # 利用三者信息，进入全连接层，得到要预测的结果
        # pred = [batch_size, output_dim]
        if self.attention is not None:
            pred = self.fc_out(torch.cat((dec_output, c, embedded), dim=1))
        else:
            pred = self.fc_out(torch.cat((dec_output, embedded), dim=1))

        # 返回预测的结果和输出给下一步的隐藏层状态
        return pred, dec_hidden.squeeze(0)


class seq2seq_main(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        '''

        :param src: 输入的血糖序列
        :param trg: 参考的输出序列，为x最后一个值和y的前5个值组成
        :param teacher_forcing_ratio:
        :return:
        '''
        # src = [batch_size, seq_len,m]   # 输入的血糖序列
        # trg = [batch_size, trg_len]   # 参考的输出序列
        # teacher_forcing_ratio is probability to use teacher forcing

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder的输出
        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer

        # enc_output= [src_len, batch_size, hid_dim * num_directions]
        # s = [batch_size, dec_hid_dim]
        enc_output, s = self.encoder(src)

        # 设置初始序列
        # first input to the decoder is the <sos> tokens
        # dec_input=[batch_size,1]
        # print('trg.shape:',trg.shape)
        dec_input = trg[:, 0:1]
        # print('dec_input', dec_input.shape)
        # 循环预测下一个序列
        for t in range(1, trg_len):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            # dec_output=[batch_size,1]
            # s=[batch_size, dec_hid_dim]
            dec_output, s = self.decoder(dec_input, s, enc_output)

            # place predictions in a tensor holding predictions for each token
            outputs[t - 1] = dec_output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = dec_output

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = trg[:, t:t + 1] if teacher_force else top1

        dec_output, s = self.decoder(dec_input, s, enc_output)
        outputs[-1] = dec_output
        return outputs.transpose(0, 1).squeeze(2)


class GRU_Attention(nn.Module):
    def __init__(self, args, rnn_layers=1, bidirection=False):
        super().__init__()
        self.use_cuda = True  # 是否使用cuda
        self.m = args.m  # 数据的维度,500
        self.input_dim = 1
        self.hidR = 256;  # number of RNN hidden units，100
        self.bidirection = bidirection
        self.num_directions = 2 if self.bidirection else 1
        self.rnn_layers = rnn_layers
        self.output_len = args.output_len
        self.output_dim = self.m
        self.dropout = nn.Dropout(p=0.2);

        self.num_directions = 2 if bidirection else 1
        self.rnn = nn.GRU(input_size=self.m,
                          hidden_size=self.hidR,
                          num_layers=1,
                          dropout=0,
                          bidirectional=bidirection);  # hidC为输入的单个序列的维度

        self.att_output = nn.Linear(self.hidR * self.num_directions * 2, 128)
        self.output = nn.Linear(128, self.output_len)
        self.tanh = nn.Tanh()

        self.w_omega = nn.Parameter(torch.Tensor(self.hidR * self.num_directions, self.hidR * self.num_directions))
        self.u_omega = nn.Parameter(torch.Tensor(self.hidR * self.num_directions, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        if self.use_cuda:
            self.rnn.cuda()
            self.att_output.cuda()
            self.output.cuda()
            self.w_omega.cuda()
            self.u_omega.cuda()

    def attention_layer(self, input):
        '''
        attention公式
        u1=tanh(W*input+b)
        a=softmax(u1u2)
        :param input: (seq_len,batch_size,hidden_size*direction
        :return:
        '''

        input = input.permute(1, 0, 2).contiguous()  # input=(batch_size,seq_len,hidden_size*direction)
        u = torch.tanh(torch.matmul(input, self.w_omega))  # u=(batch_size,seq_len,hidden_size*direction]
        att = torch.matmul(u, self.u_omega)  # att= (batch_size,seq_len,1)
        att_score = F.softmax(att, dim=1)  # att_score= (batch_size,seq_len,1)
        scored_x = input * att_score  # [batch, seq_len, hidden_dim*direction],输入乘以权值后的结果
        # context=torch.sum(scored_x,dim=1)                 # context=(batch_size,hidden_size*directions]
        return scored_x

    def forward(self, x):
        '''
        :param x: (batch_size,seq_len,m)
        :return:
        '''
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.permute(1, 0, 2).contiguous();
        outputs, h = self.rnn.forward(x)
        # outputs形状是(seq_len,batch_size,  hiddens)
        # h形状是(nums_layers*directions,batch_size,hidden_size)

        # attention过程,获得加权后的各个序列的输出,[batch, seq_len, hidden_dim*direction]
        scored_x = self.attention_layer(outputs)

        # attention score计算结束
        context = torch.sum(scored_x, dim=1)  # (batch_size,hidden*direction)
        h = h.permute(1, 0, 2).contiguous().reshape((batch_size, -1))
        final_vector = torch.cat((context, h), dim=-1)
        outs = self.att_output(final_vector)
        outs = self.tanh(outs)
        outs = self.output(outs)
        return outs;


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, enc_hid_dim, dropout):
        super().__init__()
        self.hidR = enc_hid_dim
        self.rnn = nn.GRU(input_size=input_dim,
                          hidden_size=enc_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hid_dim, output_dim)

    def forward(self, x):
        '''
        :param x: src=[batch_size,seq_len,m]
        :return:
        '''
        # x=[src_len,batch_size,m]
        x = x.permute(1, 0, 2).contiguous();
        _, out = self.rnn(x)
        out = out.view(-1, self.hidR)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_dim, enc_hid_dim)
        self.linear_1 = nn.Linear(enc_hid_dim, 32)
        self.activation_1 = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(32, 1)
        self.activation_2 = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(1, 0, 2).contiguous();
        _, out = self.rnn(x)
        out = self.dropout(torch.squeeze(out, 0));  # r=(batchsize,100)
        out = self.linear_1(out)
        out = self.dropout(out)
        out = self.activation_1(out)
        out = self.linear_2(out)
        out = self.activation_2(out)
        return out


class Discriminator_CNN(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dropout):
        super().__init__()
        self.cnn1 = nn.Conv1d(1, 100, kernel_size=3)
        self.cnn2 = nn.Conv1d(1, 100, kernel_size=3)
        self.cnn3 = nn.Conv1d(1, 100, kernel_size=3)
        self.linear_1 = nn.Linear(enc_hid_dim, 32)
        self.activation_1 = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(32, 1)
        self.activation_2 = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(1, 0, 2).contiguous();
        _, out = self.rnn(x)
        out = self.dropout(torch.squeeze(out, 0));  # r=(batchsize,100)
        out = self.linear_1(out)
        out = self.dropout(out)
        out = self.activation_1(out)
        out = self.linear_2(out)
        out = self.activation_2(out)
        return out
