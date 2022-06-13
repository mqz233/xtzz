import torch
import numpy as np
import pandas as pd
import ast
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time
from DataOperator.jsonOperator import jsonOperator as jo


class PRELSTM:
    def __init__(self, epochs=100, lr=0.01, lookback=2):
        self.epochs = epochs
        self.lr = lr
        self.lookback = lookback

    def PRELSTM(self):
        path = '../test/output.json'
        datas = jo().convertJsonToDict(path)
        xlist = datas['data_x']
        ylist = datas['data_y']

        data_x = pd.DataFrame(xlist)
        data_y = pd.DataFrame(ylist)

        col = []
        for i in range(5852):
            col.append(str(i))
        data_x.columns = col

        co = []
        for i in range(22):
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
            print(stock_y)

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

        input_dim = 5852
        hidden_dim = 32
        num_layers = 2
        output_dim = 22
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

        # for i in range(len(predict)):
        #     predict[i] = int(predict[i][0])
        #     original[i] = int(original[i][0])

        hist = hist.tolist()

        print(predict)
        print(original)
        print(hist)

        return predict, original, hist
        # 测试集损失hist_test
        # 测试集预测标签predict 测试集原标签original
        # return predict ,original ,hist

PRELSTM().PRELSTM()

