import torch.nn.functional as F
import json
import torch
import numpy as np
import pandas as pd
import ast
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time
from DataOperator.fluxdbOperator import fluxdbOperator

class POSREG:
    def __init__(self, measurement,seed=777, batch_size=512, weight_decay=0.001, nhid=128, sample_neighbor=True,
                 sparse_attention=True,structure_learning=True, pooling_ratio=0.8, dropout_ratio=0.0, lamb=1.0, device='cpu', patience=100,
                 epochs=100, lookback=50, lr=0.01):
        self.measurement = measurement
        self.seed = seed
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.nhid = nhid
        self.sample_neighbor = sample_neighbor
        self.sparse_attention = sparse_attention
        self.structure_learning = structure_learning
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio
        self.lamb = lamb
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.num_classes = 0
        self.num_features = 0
        self.lookback=lookback

    def dataset(self):
        result = fluxdbOperator().select_num_battle(self.measurement)

        fxlist = []
        ylist = []
        dic = {
            'A1': 0,
            'A2': 1,
            'A3': 2,
            'A4': 3,
            'B1': 4,
            'B2': 5,
            'B3': 6,
            'B4': 7,
            'C1': 8,
            'C2': 9,
            'C3': 10,
            'C4': 11,
            'D1': 12,
            'D2': 13,
            'D3': 14,
            'D4': 15
        }

        for i in range(len(result)):
            list = []
            xlist = []
            data = result[i]
            for key in dic.keys():
                pos = data['stage'] + data['eval']
                if pos == key:
                    ylist.append(dic[key])

            del data['sences']
            del data['frameId']
            del data['time']
            del data['Time']
            del data['name']
            # del data['svv']

            del data['stage']
            del data['eval']

            del data['radarList']
            del data['locked']
            del data['det_pro']
            del data['range_acc']
            del data['angle_acc']
            del data['atkList']
            del data['conList']
            del data['comm']
            del data['suppressList']
            del data['echo']
            del data['isRed']
            del data['type']
            del data['value']
            del data['ra_Pro_Angle']
            del data['ra_Pro_Radius']
            del data['ra_StartUp_Delay']
            del data['ra_Detect_Delay']
            del data['ra_Process_Delay']
            del data['ra_FindTar_Delay']
            del data['ra_Rang_Accuracy']
            del data['ra_Angle_Accuracy']
            del data['MisMaxAngle']
            del data['MisMaxRange']
            del data['MisMinDisescapeDis']
            del data['MisMaxDisescapeDis']
            del data['MisMaxV']
            del data['MisMaxOver']
            del data['MisLockTime']
            del data['MisHitPro']
            del data['MisMinAtkDis']
            del data['MisNum']
            del data['EchoInitState']
            del data['EchoFackTarNum']
            del data['EchoDis']
            del data['SupInitState']
            del data['SupTarNum']
            del data['SupMinDis']
            del data['SupMaxAngle']
            del data['comNum']
            del data['suppressNum']
            del data['echoNum']


            for key in data.keys():
                data[key] = json.loads(data[key])
                data[key] = np.array(data[key]).flatten().tolist()
                list.append(data[key])
            for j in range(len(list)):
                for k in range(len(list[j])):
                    xlist.append(list[j][k])
            fxlist.append(xlist)

        return fxlist, ylist

    def posreg(self):
        xlist,ylist = self.dataset()
        data_x = pd.DataFrame(xlist)
        data_y = pd.DataFrame(ylist)
        # print(data_y)

        col = []
        # print(xlist)
        for i in range(len(xlist[0])):
            col.append(str(i))
        data_x.columns = col

        # 缩放
        scaler = MinMaxScaler(feature_range=(-1, 1))
        for i in col:
            data_x[i] = scaler.fit_transform(data_x[i].values.reshape(-1, 1))
        data_y = scaler.fit_transform(data_y.values.reshape(-1 ,1))

        # 制作数据集
        def split_data(stock, stock_y, lookback):
            data_raw = stock.to_numpy()
            data = []
            stock_y = stock_y[::lookback]
            stock_y = np.array(stock_y)
            for i in range(len(stock_y)):
                stock_y[i] = np.array(stock_y[i])
            # print(stock_y)

            # you can free play（seq_length）
            for index in range(0,len(data_raw),lookback):
                if index+lookback <= len(data_raw):
                    data.append(data_raw[index: index + lookback])


            data = np.array(data)
            test_set_size = int(np.round(0.2 * data.shape[0]))
            train_set_size = data.shape[0] - (test_set_size)

            x_train = data[:train_set_size]
            y_train = stock_y[:train_set_size]

            x_test = data
            y_test = stock_y

            return [x_train, y_train, x_test, y_test]

        lookback = self.lookback
        x_train, y_train, x_test, y_test = split_data(data_x, data_y, lookback)
        print('x_train.shape = ', x_train.shape)
        print('y_train.shape = ', y_train.shape)
        print('x_test.shape = ', x_test.shape)
        print('y_test.shape = ', y_test.shape)

        # LSTM
        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
        y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
        # y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
        # y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

        input_dim = len(xlist[0])
        hidden_dim = 32
        num_layers = 2
        output_dim = 1
        num_epochs = self.epochs

        class LSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
                super(LSTM, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers

                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
                out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
                out = self.fc(out[:, -1, :])
                return out

        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        # 训练
        hist = np.zeros(num_epochs)
        hist_test = np.zeros(num_epochs)
        start_time = time.time()
        lstm = []

        for t in range(num_epochs):
            y_train_pred = model(x_train)

            loss = criterion(y_train_pred, y_train_lstm)
            print("Epoch ", t, "MSE: ", loss.item())
            # 训练集损失
            hist[t] = loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))

        y_test_pred = model(x_test)
        predict = np.around(scaler.inverse_transform(y_test_pred.detach().numpy())).tolist()
        original = np.around(scaler.inverse_transform(y_test_lstm.detach().numpy())).tolist()


        for i in range(len(predict)):
            predict[i] = int(predict[i][0])
            original[i] = int(original[i][0])

        original = original[:len(predict)]

        hist = hist.tolist()

        return predict, original, hist

