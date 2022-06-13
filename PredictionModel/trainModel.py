# from tensorflow_features_lstm import Lstm

import torch
import torch.nn as nn
from pylab import *
from torch.autograd import Variable
import os

from DataOperator.jsonOperator import jsonOperator as jo
from conf.readConfig import readConfig as rc
# from tensorflow_features_lstm import Lstm
from PredictionModel.features_lstm import LSTM_FEATURES

mpl.rcParams['font.sans-serif'] = ['SimHei']


class LSTM_PREDICT(object):
    def __init__(self, n_steps: int, data_path: str):
        self.n_steps = n_steps  # 训练的步长，即用几帧的数据训练下一帧的数据
        self.datas_path = data_path
        self.file_name = os.path.split(data_path)[1][:-5]
        self.lstm_type = 0 if data_path.find('static') > 0 else 1  ## 用于区分是否是静态数据预测还是动态数据预测，0表静态，1表动态

    def features_predict(self):
        features_pre = LSTM_FEATURES(self.n_steps, self.datas_path).lstm()  # 3:步长——>预测dataX本身时的步长可以随意取
        return features_pre

    def lstm(self):
        import numpy as np
        # 一、数据准备
        # datas_path = "D:\OneDrive - mail.dlut.edu.cn/桌面/ZZ项目/功能实现/代码/test_datas/dataset/json_output_5055555.json"
        datas_path = self.datas_path
        datas = jo().convertJsonToDict(datas_path)
        dataX = datas['data_x']  # 210
        dataY = datas['data_y']
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        input_size = dataX.shape[1]
        # # todo：归一化处理，这一步必不可少，不然后面训练数据误差会很大，模型没法用

        # 训练集划分
        train_size = int(len(dataX) * 1)  # 已有的数据全用来训练模型
        x_train = dataX[:train_size]  # 训练数据
        y_train = dataY[:train_size]  # 训练数据目标值
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # x_train = x_train.reshape(-1, 1, 210)  # 将训练数据调整成pytorch中lstm算法的输入维度： -1：序列数, 1：batch_size, 2:特征维数
        # y_train = y_train.reshape(-1, 1, 10)  # 将目标值调整成pytorch中lstm算法的输出维度
        x_train = x_train.reshape(-1, 1, input_size)
        y_train = y_train.reshape(-1, 1, 10)
        # 将ndarray数据转换为张量，因为pytorch用的数据类型是张量
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)


        class RNN(nn.Module):
            def __init__(self, input_size):
                super(RNN, self).__init__()  # 面向对象中的继承
                self.lstm = nn.LSTM(input_size, input_size * 2, 2)  # 输入数据2个特征维度，6个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
                self.out = nn.Linear(input_size * 2, 10)  # 线性拟合，接收数据的维度为6，输出数据的维度为1

            def forward(self, x):
                x1, _ = self.lstm(x)
                a, b, c = x1.shape
                out = self.out(x1.view(-1, c))  # 因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
                out1 = out.view(a, b, -1)  # 因为是循环神经网络，最后的时候要把二维的out调整成三维数据，下一次循环使用
                return out1

        rnn = RNN(input_size)
        # rnn.cuda(0)
        torch.nn.DataParallel(rnn)
        # 参数寻优，计算损失函数
        optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)  # torch.optim.SGD
        loss_func = nn.MSELoss()
        train_loss = []
        # error_curve_list = []
        epoch = 120
        for i in range(epoch):
            var_x = Variable(x_train).type(torch.FloatTensor).cuda(0)
            var_y = Variable(y_train).type(torch.FloatTensor).cuda(0)  # Tensor：3
            out = rnn(var_x)  # 输出的预测y值：Tensor：3
            loss = loss_func(out, var_y)
            # error_curve_list.append(out-var_y)
            optimizer.zero_grad()  # 梯度初始化为0，因为一个batch的loss关于weight的导数，是所有sample的loss关于weight导数的累加和
            loss.backward()  # 反向传播求梯度
            optimizer.step()  # 更新所有参数
            train_loss.append(loss.item())
            if (i + 1) % 10 == 0:
                print('Epoch:{}, Loss:{:.5f}'.format(i + 1, loss.item()))
        plt.figure()
        plt.xlabel("epoches", fontsize=16)
        plt.ylabel("loss", fontsize=16)
        plt.plot(range(1, epoch), train_loss[1:], label="Train_Loss")
        plt.savefig(os.path.join(rc().getImagePath(), self.file_name + r'loss.png'))
        plt.show()

        # torch.save(rnn.state_dict(), "LSTM_svv.pth")  # 保存模型

        # 四、模型测试
        # 准备测试数据
        dataX = np.array(dataX)
        # 添加预测的pre_dataX
        pre_dataX = array(self.features_predict())
        all_dataX = np.append(dataX, pre_dataX)

        dataX1 = all_dataX.reshape(-1, 1, input_size)
        dataX2 = torch.from_numpy(dataX1)
        var_dataX = Variable(dataX2).type(torch.FloatTensor).cuda(0)
        pred = rnn(var_dataX)
        # pred_test = pred.view(-1).data.numpy()  # 转换成一维的ndarray数据，这是预测值 ,dataY为真实值
        pred_test = pred.view(-1, 10).data.numpy()  # 变为-1*10的矩阵，再求和第二维，实现每隔10个标签求和
        sum_y = dataY.sum(axis=1)
        sum_pred = pred_test.sum(axis=1)

        # 五、画图检验
        # plt.plot(pred.view(-1).data.numpy(), 'r', label='prediction')
        # plt.plot(dataY.reshape(10000,), 'b', label='real')
        plt.figure()
        plt.xlabel("帧数")  # x轴上的名字
        plt.ylabel("平台存活数")  # y轴上的名字
        if type:
            plt.title('静态-动态混合ZZ数据预测结果')
        else:
            plt.title('静态ZZ数据预测结果')
        plt.plot(sum_pred[1:], 'r', label='prediction')
        plt.plot(sum_y, 'b', label='real')
        plt.legend(loc='best')
        plt.savefig(os.path.join(rc().getImagePath(), self.file_name + r'.png'))
        plt.show()


def readPredictedData(war_name: str, type: bool = 1):
    # todo 添加读取 Image 的函数
    location = rc().getImagePath()
    war_name = war_name + '_dynamic.json' if type == '1' else war_name + '_static.json'
    warDatasetDir = os.path.join(location, war_name)
    imageData = jo().convertJsonToDict(warDatasetDir)
    return imageData['x'], imageData['y']


# result = LSTM_PREDICT("..\data\json_output_5055555_static.json").lstm()
# result = LSTM_PREDICT(3,  "E:\OneDrive - mail.dlut.edu.cn\桌面\djangoProject2\PredictionModel\data\dataset\dataset\json_output_5055555_dynamic.json").lstm()

# predictionSvv('json_output_5055555',1) ## 输入案例
