import torch
import torch.nn as nn
from pylab import *
from torch.autograd import Variable

from DataOperator.jsonOperator import jsonOperator as jo
import os

# import numpy as np

mpl.rcParams['font.sans-serif'] = ['SimHei']


class LSTM_FEATURES(object):
    def __init__(self, n_steps: int, data_path: str, frame: int):
        self.frame = frame
        self.n_steps = n_steps
        self.datas_path = data_path
        self.file_name = os.path.split(data_path)[1][:-5]
        self.lstm_type = 0 if data_path.find('static') > 0 else 1  ## 用于区分是否是静态数据预测还是动态数据预测，0表静态，1表动态

    def split_sequence(self, sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    def lstm(self):
        import numpy as np
        # 一、数据准备
        # datas_path = "D:\OneDrive - mail.dlut.edu.cn/桌面/ZZ项目/功能实现/代码/test_datas/dataset/json_output_5055555.json"
        datas_path = self.datas_path
        datas = jo().convertJsonToDict(datas_path)
        data = datas['data_x']  # 210 预测属性值
        data = data[:self.frame+2] #起码两个初始data值，才能构成1个x，1个y，所以是frame+1 (frame从1开始取值)
        # dataY = datas['data_y']

        dataX, dataY = self.split_sequence(data, self.n_steps)

        x_train = np.array(dataX)
        y_train = np.array(dataY)
        input_size = x_train.shape[2]
        num_data = dataX.shape[0]
        # # todo：归一化处理，这一步必不可少，不然后面训练数据误差会很大，模型没法用
        # n_steps个x预测一个y
        y_train = y_train.reshape(-1, 1, input_size)
        # 将ndarray数据转换为张量，因为pytorch用的数据类型是张量
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)

        class RNN(nn.Module):
            def __init__(self, input_size):
                super(RNN, self).__init__()  # 面向对象中的继承
                self.lstm = nn.LSTM(input_size, input_size * 2, 2)  # 输入数据2个特征维度，6个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
                self.out = nn.Linear(input_size * 2, input_size)  # 线性拟合，接收数据的维度为6，输出数据的维度为1

            def forward(self, x):
                x1, _ = self.lstm(x)
                a, b, c = x1.shape
                out = self.out(x1.view(-1, c))  # 因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
                out1 = out.view(a, b, -1)  # 因为是循环神经网络，最后的时候要把二维的out调整成三维数据，下一次循环使用
                return out1

        rnn = RNN(input_size)

        # # # 训练模型
        # rnn.cuda(0)
        # optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)  # torch.optim.SGD
        # loss_func = nn.MSELoss()
        # train_loss = []
        # # error_curve_list = []
        # for i in range(10):
        #     var_x = Variable(x_train).type(torch.FloatTensor).cuda(0)
        #     var_y = Variable(y_train).type(torch.FloatTensor).cuda(0)  # Tensor：3
        #     out = rnn(var_x)  # 输出的预测y值：Tensor：3
        #     loss = loss_func(out, var_y)
        #     # error_curve_list.append(out-var_y)
        #     optimizer.zero_grad()  # 梯度初始化为0，因为一个batch的loss关于weight的导数，是所有sample的loss关于weight导数的累加和
        #     loss.backward()  # 反向传播求梯度
        #     optimizer.step()  # 更新所有参数
        #     train_loss.append(loss.item())
        #     if (i + 1) % 2 == 0:
        #         print('Epoch:{}, Loss:{:.5f}'.format(i + 1, loss.item()))
        # # demonstrate prediction
        # torch.save(rnn.state_dict(), "n_steps1000_STM_features.pth")#保存模型

        rnn.load_state_dict(torch.load("PredictionModel\LSTM_features.pth"))  # 加载模型
        rnn.cuda(0)

        x_input = array(dataX[num_data - self.n_steps:, :, :])
        x_input = x_input.reshape((-1, self.n_steps, input_size))
        dataX2 = torch.from_numpy(x_input)
        var_dataX = Variable(dataX2).type(torch.FloatTensor)
        var_dataX = var_dataX.cuda(0)
        yhat_list = []
        for i in range(1000 - self.frame):
            pred = rnn(var_dataX)
            # pred_test = pred.view(-1).data.numpy()  # 转换成一维的ndarray数据，这是预测值 ,dataY为真实值
            pred_test = torch.Tensor.cpu(pred.view(-1, input_size).data).numpy()#将显存中的tensor数据复制到内存中，后转化为numpy
            # yhat = model.predict(x_input, verbose=0)
            yhat = array(pred_test)[0]
            x_input = np.append(x_input, yhat)[input_size:]
            x_input = x_input.reshape((-1, self.n_steps, input_size))
            yhat_list.append(list(yhat))
        print(array(yhat_list).shape)
        return yhat_list

# result = LSTM_PREDICT(3,"..\data\json_output_5055555_static.json").lstm()
# result = LSTM_FEATURES(2, "E:\项目数据集\dataset\json_output_5055555_dynamic.json",1).lstm()


# predictionSvv('json_output_5055555',1) ## 输入案例
