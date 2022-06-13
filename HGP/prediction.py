import torch.nn.functional as F
import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv
from HGP.layers import GCN, HGPSLPool
from HGP.mtm_1_reg import StaticGraphTemporalSignal, MTMDatasetLoader
import numpy as np
import pandas as pd
import ast
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time

###修改预训练模型
# pre_model = "./84.pth"
# dict = torch.load(pre_model)
# for key in list(dict.keys()):
#     # print(key)
#     if key.startswith('lin'):
#         del dict[key]
# torch.save(dict, './model_deleted.pth')
# # # #验证修改是否成功
# changed_dict = torch.load('./model_deleted.pth')
# for key in dict.keys():
#     print(key)


class MyModel(torch.nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        return x

def dataset_split(dataset,train_ratio):
    train_snapshots = int(train_ratio * dataset.snapshot_count)
    train_snapshots = StaticGraphTemporalSignal(
            dataset.edge_index,
            dataset.edge_weight,
            dataset.features[0:train_snapshots],
            dataset.targets[0:train_snapshots],
            **{key: getattr(dataset, key)[0:train_snapshots] for key in dataset.additional_feature_keys}
        )
    return train_snapshots


class MTMPRE:
    def __init__(self, seed=777, batch_size=512, weight_decay=0.001, nhid=128, sample_neighbor=True,
                 sparse_attention=True,structure_learning=True, pooling_ratio=0.8, dropout_ratio=0.0, lamb=1.0, device='cpu', patience=100,
                 epochs=100, lookback=20, lr=0.01):
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
        self.num_classes = 6
        self.num_features = 3
        self.lookback=lookback


    def MTMPRE(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        loader = MTMDatasetLoader()
        dataset = loader.get_dataset()
        mymodel = MyModel(self).to(self.device)
        mymodel.load_state_dict(torch.load('HGP/model_deleted.pth'))  ##更改后的模型参数文件
        xlist = []
        ylist = []
        dataset = dataset_split(dataset, 0.1)   ##选多少张图，共14000+

        for i,data in enumerate(dataset):
            data = data.to(self.device)
            x = mymodel(data)
            numpyx = x.detach().numpy()[0]
            list = []
            for xx in numpyx:
                list.append(float(xx))
            xlist.append(list)
            ylist.append(data.y.numpy()[0])

        data_x = pd.DataFrame(xlist)

        col = []
        for i in range(256):
            col.append(str(i))
        data_x.columns = col

        data_y = pd.DataFrame(ylist)

        #缩放
        scaler = MinMaxScaler(feature_range=(-1, 1))
        for i in col:
            data_x[i] = scaler.fit_transform(data_x[i].values.reshape(-1,1))
        data_y = scaler.fit_transform(data_y.values.reshape(-1,1))

        #制作数据集
        def split_data(stock, stock_y,lookback):
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
        x_train, y_train, x_test, y_test = split_data(data_x,data_y, lookback)
        print('x_train.shape = ',x_train.shape)
        print('y_train.shape = ',y_train.shape)
        print('x_test.shape = ',x_test.shape)
        print('y_test.shape = ',y_test.shape)

        #LSTM
        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
        y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
        # y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
        # y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

        input_dim = 256
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

        #训练
        hist = np.zeros(num_epochs)
        hist_test = np.zeros(num_epochs)
        start_time = time.time()
        lstm = []

        for t in range(num_epochs):
            y_train_pred = model(x_train)

            loss = criterion(y_train_pred, y_train_lstm)
            print("Epoch ", t, "MSE: ", loss.item())
            #训练集损失
            hist[t] = loss.item()


            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))

        # torch.save(model.state_dict(), './lstm.pth')

        y_test_pred = model(x_test)
        predict = np.around(scaler.inverse_transform(y_test_pred.detach().numpy())).tolist()
        original = np.around(scaler.inverse_transform(y_test_lstm.detach().numpy())).tolist()

        for i in range(len(predict)):
            predict[i] = int(predict[i][0])
            original[i] = int(original[i][0])

        hist = hist.tolist()

        # print(predict)
        # print(original)
        # print(hist)
        # 测试集损失hist_test
        # 测试集预测标签predict 测试集原标签original
        return predict,original,hist


# MTMPRE().MTMPRE()
# print(len(predict))
# print(predict)
# #训练集曲线数据
# predict = pd.DataFrame(np.around(scaler.inverse_transform(y_train_pred.detach().numpy())))
# original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))
#
#
# #训练过程绘图
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set_style("darkgrid")
#
# fig = plt.figure()
# fig.subplots_adjust(hspace=0.2, wspace=0.2)
#
# plt.subplot(1, 3, 1)
# ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
# ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (LSTM)", color='tomato')
# ax.set_title('train', size = 14, fontweight='bold')
# ax.set_xlabel("frame", size = 14)
# ax.set_ylabel("label", size = 14)
# ax.set_xticklabels('', size=10)
#
#
# plt.subplot(1, 3, 2)
# ax = sns.lineplot(data=hist, color='royalblue')
# ax.set_xlabel("Epoch", size = 14)
# ax.set_ylabel("Loss", size = 14)
# ax.set_title("Training Loss", size = 14, fontweight='bold')
# fig.set_figheight(6)
# fig.set_figwidth(16)
#
# #测试集曲线数据
# test_predict = model(x_test)
# test_predict = pd.DataFrame(np.around(scaler.inverse_transform(test_predict.detach().numpy())))
# test_original = pd.DataFrame(scaler.inverse_transform(y_test_lstm.detach().numpy()))
#
# plt.subplot(1, 3, 3)
# ax = sns.lineplot(x = test_original.index, y = test_original[0], label="Data", color='royalblue')
# ax = sns.lineplot(x = test_predict.index, y = test_predict[0], label="Testing Prediction (LSTM)", color='tomato')
# ax.set_title('test', size = 14, fontweight='bold')
# ax.set_xlabel("frame", size = 14)
# ax.set_ylabel("label", size = 14)
# ax.set_xticklabels('', size=10)
# plt.show()


