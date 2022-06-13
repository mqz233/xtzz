import os

import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from numpy import array

from DataOperator.jsonOperator import jsonOperator as jo


# import numpy as np

# mpl.rcParams['font.sans-serif'] = ['SimHei']


class Lstm(object):
    def __init__(self, data_path: str, n_steps):
        self.datas_path = data_path
        self.file_name = os.path.split(data_path)[1][:-5]
        self.lstm_type = 0 if data_path.find('static') > 0 else 1  ## 用于区分是否是静态数据预测还是动态数据预测，0表静态，1表动态
        self.n_steps = n_steps

    # 滑动窗口生成数据
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
        datas_path = self.datas_path
        datas = jo().convertJsonToDict(datas_path)
        data = datas['data_x']  # 210 预测属性值
        # dataY = datas['data_y']

        dataX, dataY = self.split_sequence(data, self.n_steps)
        # dataY = self.split_sequence(dataY,3)[0]
        # dataX = np.array(dataX)
        # dataY = np.array(dataY)[1:]
        input_size = dataX.shape[2]
        num_data = dataX.shape[0]
        # # todo：归一化处理，这一步必不可少，不然后面训练数据误差会很大，模型没法用

        # 训练集划分

        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(self.n_steps, input_size)))  # 隐藏层，输入，n_steps 特征维
        model.add(Dense(input_size))  # 加入全连接层
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(dataX, dataY, epochs=100, batch_size=1, verbose=2)  # 迭代次数，批次数，verbose决定是否显示每次迭代
        # demonstrate prediction
        x_input = array(dataX[num_data - self.n_steps:, :, :])  # 输入最后n_steps行数据 去预测下一个数据
        x_input = x_input.reshape((-1, self.n_steps, input_size))  # 中间参数：n_steps
        yhat_list = []
        # todo:最后一帧不一定是1200帧，可能需要修改
        for i in range(1200 - (num_data + self.n_steps)):  # num_data == 样本总数-n_steps
            yhat = model.predict(x_input, verbose=0)
            yhat = array(yhat)[0]
            x_input = np.append(x_input, yhat)[input_size:]
            x_input = x_input.reshape((-1, self.n_steps, input_size))  # 中间参数：n_steps
            yhat_list.append(list(yhat))
        shape = array(yhat_list).shape
        # print(array(yhat_list).shape)
        # 返回预测的属性值列表
        return yhat_list

# result = Lstm("json_output_5055555_static.json",3).lstm()
# result = Lstm("..\data\json_output_5055555_dynamic.json",3).lstm()


# predictionSvv('json_output_5055555',1) ## 输入案例
