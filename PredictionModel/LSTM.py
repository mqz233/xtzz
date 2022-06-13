import torch
import numpy as np
import pandas as pd
import ast
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time
from DataOperator.jsonOperator import jsonOperator as jo
from PredictionModel.create_dataset import Createdataset

class PRELSTM:
    def __init__(self, measurement, epochs=100, lr=0.01, lookback=20):
        self.measurement = measurement
        self.epochs = epochs
        self.lr = lr
        self.lookback = lookback

    def PRELSTM(self):
        dataset = Createdataset(self.measurement)
        xlist,ylist = dataset.createdataset2()

        data_x = pd.DataFrame(xlist)
        data_y = pd.DataFrame(ylist)
        # print(data_y)

        col = []
        # print(xlist)
        for i in range(len(xlist[0])):
            col.append(str(i))
        data_x.columns = col

        co = []
        for i in range(len(ylist[0])):
            co.append(str(i))
        data_y.columns = co

        # 缩放
        scaler = MinMaxScaler(feature_range=(-1, 1))
        for i in col:
            data_x[i] = scaler.fit_transform(data_x[i].values.reshape(-1, 1))

        # for i in co:
        #     data_y[i] = scaler.fit_transform(data_y[i].values.reshape(-1, 1))
        # data_y = scaler.fit_transform(data_y.values.reshape(-1 ,1))

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
        output_dim = len(ylist[0])
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

        # torch.save(model.state_dict(), 'LSTM.pth')

        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))

        y_test_pred = model(x_test)

        predict = np.around(y_test_pred.detach().numpy()).tolist()
        original = np.around(y_test_lstm.detach().numpy()).tolist()

        # for i in range(len(predict)):
        #     predict[i] = int(predict[i][0])
        #     original[i] = int(original[i][0])

        hist = hist.tolist()

        # print(predict)
        # print(original)

        isred = dataset.red_blue()
        red1 = []  #实际
        red2 = []  #下一帧
        red3 = []  #结果
        red4 = []  #第一帧
        blue1 = []
        blue2 = []
        blue3 = []
        blue4 = []
        r3 = 0
        r4 = 0
        b3 = 0
        b4 = 0
        for j in range(len(isred)):
            if isred[j] ==1 :
                r3 += original[-1][j]
                r4 += original[0][j]
            else:
                b3 += original[-1][j]
                b4 += original[0][j]

        for i in range(len(predict)):
            red1.append(0)
            red2.append(0)
            red3.append(r3)
            red4.append(r4)
            blue1.append(0)
            blue2.append(0)
            blue3.append(b3)
            blue4.append(b4)
            for j in range(len(isred)):
                if isred[j] == 1:
                    red1[i] += original[i][j]
                    red2[i] += predict[i][j]
                else:
                    blue1[i] += original[i][j]
                    blue2[i] += predict[i][j]

        # print(red1)
        # print(hist)


        return red1, red2, red3, red3, blue1, blue2, blue3, blue3, hist
        # 测试集损失hist_test
        # 测试集预测标签predict 测试集原标签original
        # return predict ,original ,hist

# PRELSTM('planetest1284').PRELSTM()

