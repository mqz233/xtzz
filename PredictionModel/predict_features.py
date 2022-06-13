import numpy as np
import torch
import torch.nn as nn
from pylab import *
from torch.autograd import Variable
import os
from DataOperator.jsonOperator import jsonOperator as jo

mpl.rcParams['font.sans-serif'] = ['SimHei']

class LSTM_FEATURES(object):
    def __init__(self, data_path: str, frame: int ,n_steps=1000):
        self.frame = frame
        self.n_steps = n_steps
        self.datas_path = data_path
        self.file_name = os.path.split(data_path)[1][:-5]
        self.lstm_type = 0 if data_path.find('static') > 0 else 1  ## 用于区分是否是静态数据预测还是动态数据预测，0表静态，1表动态

    def split_sequence(self, sequence, n_steps):
        X, y = np.array([]), list()

        for i in range(len(sequence)):
            num_zeros = n_steps - i-1
            x_size = np.array(sequence[0]).shape[0] #随意取一个，得到input_size=210/110
            # one_zeros = np.zeros(1,x_size)
            new_x =  np.append(np.zeros((num_zeros,x_size)),np.array(sequence[:i+1]))
            X = np.append(X,new_x)
            y.append(sequence[i+1])      ## x predict x+1
            if i == len(sequence)-2:
                break
        X = X.reshape(-1,n_steps,shape(y)[1])
        return np.array(X[:,:,:]), np.array(y)

    def lstm(self):
        import numpy as np
        # 一、数据准备
        # datas_path = "D:\OneDrive - mail.dlut.edu.cn/桌面/ZZ项目/功能实现/代码/test_datas/dataset/json_output_5055555.json"
        datas_path = self.datas_path
        datas = jo().convertJsonToDict(datas_path)
        data = datas['data_x']  # 210 预测属性值
        # dataY = datas['data_y']
        # data = data[:self.frame+1]
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
                self.lstm = nn.LSTM(input_size, 50, 2)  # 输入数据2个特征维度，6个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
                self.out = nn.Linear(50, input_size)  # 线性拟合，接收数据的维度为6，输出数据的维度为1

            def forward(self, x):
                x1, _ = self.lstm(x)
                a, b, c = x1.shape
                out = self.out(x1.view(-1, c))  # 因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
                out1 = out.view(a, b, -1)  # 因为是循环神经网络，最后的时候要把二维的out调整成三维数据，下一次循环使用
                return out1

        rnn = RNN(input_size)
        # rnn.cuda(0)
        device_ids = [0, 1]
        rnn = torch.nn.DataParallel(rnn, device_ids=device_ids).cuda()#多个显卡训练模型
        # # 训练模型
        optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)  # torch.optim.SGD
        loss_func = nn.MSELoss()
        train_loss = []
        # error_curve_list = []
        for i in range(20):
            var_x = Variable(x_train).type(torch.FloatTensor).cuda(0)
            var_y = Variable(y_train).type(torch.FloatTensor).cuda(0)  # Tensor：3
            out = rnn(var_x)  # 输出的预测y值：Tensor：3
            loss = loss_func(out, var_y)
            # error_curve_list.append(out-var_y)
            optimizer.zero_grad()  # 梯度初始化为0，因为一个batch的loss关于weight的导数，是所有sample的loss关于weight导数的累加和
            loss.backward()  # 反向传播求梯度
            optimizer.step()  # 更新所有参数
            train_loss.append(loss.item())
            if (i + 1) % 2 == 0:
                print('Epoch:{}, Loss:{:.5f}'.format(i + 1, loss.item()))
        # # demonstrate prediction
        # torch.save(rnn.state_dict(), "steps1000_LSTM_features.pth")#保存模型


# result = LSTM_PREDICT(3,"..\data\json_output_5055555_static.json").lstm()
# result = LSTM_FEATURES( "../PredictionModel/data/dataset/dataset/json_output_5055555_dynamic.json",1000).lstm() #todo:frame参数改为检测文件夹下的帧数


# predictionSvv('json_output_5055555',1) ## 输入案例
