import os

import torch
import glob
import numpy as np
import pandas as pd
import ast
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time
from DataOperator.jsonOperator import jsonOperator as jo
from PredictionModel.trainset import Createtrainset


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

# 制作数据集
def split_data(stock, stock_y, lookback):
    data_raw = stock.to_numpy()
    data = []
    stock_y = stock_y[lookback:]
    stock_y = np.array(stock_y)
    for i in range(len(stock_y)):
        stock_y[i] = np.array(stock_y[i])
    # print(stock_y)

    # you can free play（seq_length）
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data)

    x_train = data
    y_train = stock_y

    return [x_train, y_train]

class TrainLSTM:
    def __init__(self, tag, epochs=100, lr=0.01, lookback=20):
        self.dataset = Createtrainset(tag)
        self.tag = tag
        self.epochs = epochs
        self.lr = lr
        self.lookback = lookback

    def trainLSTM(self):
        dataset = self.dataset
        trainset = dataset.createtrainset()
        setnum = len(trainset)

        input_dim = len(trainset[0]['data_x'][0])
        hidden_dim = 32
        num_layers = 2
        output_dim = len(trainset[0]['data_y'][0])

        num_epochs = self.epochs
        for i in range(setnum):
            xlist = trainset[i]['data_x']
            ylist = trainset[i]['data_y']

            data_x = pd.DataFrame(xlist)
            data_y = pd.DataFrame(ylist)
            # print(data_y)

            col = []
            # print(xlist)
            for ii in range(len(xlist[0])):
                col.append(str(i))
            data_x.columns = col


            co = []
            for ii in range(len(ylist[0])):
                co.append(str(i))
            data_y.columns = co

            # 缩放
            scaler = MinMaxScaler(feature_range=(-1, 1))
            for i in col:
                data_x[i] = scaler.fit_transform(data_x[i].values.reshape(-1, 1))

            # for i in co:
            #     data_y[i] = scaler.fit_transform(data_y[i].values.reshape(-1, 1))
            # data_y = scaler.fit_transform(data_y.values.reshape(-1 ,1))

            x_train, y_train = split_data(data_x, data_y, self.lookback)

            print('x_train.shape = ', x_train.shape)
            print('y_train.shape = ', y_train.shape)

            # LSTM
            x_train = torch.from_numpy(x_train).type(torch.Tensor)
            y_train = torch.from_numpy(y_train).type(torch.Tensor)

            trainset[i]['data_x'] = x_train
            trainset[i]['data_y'] = y_train

            # y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
            # y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        # 训练
        hist = np.zeros(num_epochs)
        hist_test = np.zeros(num_epochs)
        start_time = time.time()
        lstm = []

        for t in range(num_epochs):
            loss_train = 0.0
            for i in range(setnum):
                x_train = trainset[i]['data_x']
                y_train = trainset[i]['data_y']

                y_train_pred = model(x_train)

                loss = criterion(y_train_pred, y_train)

                loss_train += loss.item()

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            # 训练集损失
            hist[t] = loss_train
            print("Epoch ", t, "MSE: ", loss_train)

        path = 'PredictionModel/model/{}'.format(self.tag)
        isExists = os.path.exists(path)
        if isExists:
            pth = glob.glob(path+'/*.pth')
            cnt = len(pth)
            torch.save(model.state_dict(), path+'/{}.pth'.format(str(cnt) + '-in'))
        else:
            os.makedirs(path)
            torch.save(model.state_dict(), path + '/{}.pth'.format(str(0) + '-in'))

        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))
        return hist.tolist()


    def trainLSTMno(self):
        dataset = self.dataset
        trainset = dataset.createtrainsetno()
        setnum = len(trainset)

        input_dim = len(trainset[0]['data_x'][0])
        hidden_dim = 32
        num_layers = 2
        output_dim = len(trainset[0]['data_y'][0])

        num_epochs = self.epochs
        for i in range(setnum):
            xlist = trainset[i]['data_x']
            ylist = trainset[i]['data_y']

            data_x = pd.DataFrame(xlist)
            data_y = pd.DataFrame(ylist)
            # print(data_y)

            col = []
            # print(xlist)
            for ii in range(len(xlist[0])):
                col.append(str(ii))
            data_x.columns = col

            co = []
            for ii in range(len(ylist[0])):
                co.append(str(ii))
            data_y.columns = co

            # 缩放
            scaler = MinMaxScaler(feature_range=(-1, 1))
            for ii in col:
                data_x[ii] = scaler.fit_transform(data_x[ii].values.reshape(-1, 1))

            # for i in co:
            #     data_y[i] = scaler.fit_transform(data_y[i].values.reshape(-1, 1))
            # data_y = scaler.fit_transform(data_y.values.reshape(-1 ,1))

            x_train, y_train = split_data(data_x, data_y, self.lookback)

            print('x_train.shape = ', x_train.shape)
            print('y_train.shape = ', y_train.shape)

            # LSTM
            x_train = torch.from_numpy(x_train).type(torch.Tensor)
            y_train = torch.from_numpy(y_train).type(torch.Tensor)

            trainset[i]['data_x'] = x_train
            trainset[i]['data_y'] = y_train

            # y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
            # y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        # 训练
        hist = np.zeros(num_epochs)
        hist_test = np.zeros(num_epochs)
        start_time = time.time()
        lstm = []

        for t in range(num_epochs):
            loss_train = 0.0
            for i in range(setnum):
                x_train = trainset[i]['data_x']
                y_train = trainset[i]['data_y']

                y_train_pred = model(x_train)

                loss = criterion(y_train_pred, y_train)

                loss_train += loss.item()

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            # 训练集损失
            hist[t] = loss_train
            print("Epoch ", t, "MSE: ", loss_train)

        path = 'PredictionModel/model/{}'.format(self.tag)
        isExists = os.path.exists(path)
        if isExists:
            pth = glob.glob(path + '/*.pth')
            cnt = len(pth)
            torch.save(model.state_dict(), path + '/{}.pth'.format(str(cnt) + '-no'))
        else:
            os.makedirs(path)
            torch.save(model.state_dict(), path + '/{}.pth'.format(str(0) + '-no'))

        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))
        return hist.tolist()

