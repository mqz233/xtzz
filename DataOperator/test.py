# from influxdb import InfluxDBClient
# import requests
# import time
# json_body = [
#     {
#         "measurement": "BB",
#         "tags": {
#             "host": "server020",
#             "region": "us-west"
#         },
#         "time": "2009-11-10T23:00:00Z",
#         "fields": {
#             "value": 0.56
#         }
#     }
# ]
# s = requests.session()
# s.keep_alive = False
# # client = InfluxDBClient('localhost', 8086, 'root', 'root', 'LDR')  # 指定连接的数据库
# client = InfluxDBClient('210.30.96.105', 6111, 'root', 'root', 'LDR')  # 指定连接的数据库
# # client = InfluxDBClient('127.0.0.1', 8086, 'root', 'root', 'LDR') # timeout 超时时间 10秒
# # client.write_points(json_body)
# result = client.query('select * from XC;')
# print('获取数据库列表：')
#
# database_list = client.get_list_database()
#
# print(database_list)
#
# print(result)
# import tensorflow
# from multiprocessing import Process
# import time
# from PredictionModel.figs import pre_result
# import os

# class Myprocess(Process):
#     def __init__(self,name):
#         super().__init__()
#         self.name = name
#         self.result = name
#
#     def run(self):
#         print('%s is runing' % self.name)
#         time.sleep(1)
#         print('%s is done' % self.name)
#         self.result = 'wwe'
#         return self.result
#
#
# if __name__ == '__main__':
#     p1 = Myprocess('aaa')
#     # p2= Myprocess(target=att, args=('xiaojiu',))
#     p1.run()
#     p1.start()
#     print("zzz",p1.result)
#     p1.join() #这里xiaojiu 执行完毕之后才会执行其他进程
#     print('主进程')
# from DataOperator.createDataSet import createDataSet
# createDataSet = createDataSet()
# A = createDataSet.get_dynamic_data_list("..\data\push")
#
# print(len(A))

# 查询influxdb动态数据
# from DataOperator.createDataSet import createDataSet
# print(createDataSet().obtainDynamicDataFromDatabase("XC")[0].values())
# print(createDataSet().obtainDynamicDataFromDatabase("xxx_001"))
#

# from pylab import *
# import json
# import os
# from conf.readConfig import readConfig
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# getpoint = '5255555' #从前端接受到的预测场次
# result_predict_root_path = readConfig().getResultPredictRootPath()
# result_predict_path = os.path.join(result_predict_root_path,getpoint,'red_blue_dynamic.json')
#
# # with open('../data/result_predict_data/5355555/red_blue_dynamic.json', 'rb') as f:
# with open(result_predict_path, 'rb') as f:
#     params = json.load(f)
# # x  = params.get('x')
# red_y1 = params.get('red_y1')
# red_y2 = params.get('red_y2')
# red_y3 = params.get('red_y3')
# red_y4 = params.get('red_y4')
# blue_y1 = params.get('blue_y1')
# blue_y2 = params.get('blue_y2')
# blue_y3 = params.get('blue_y3')
# blue_y4 = params.get('blue_y4')
# x = [i for i in range(998)]
#
# plt.figure()
# plt.xlabel("帧数")  # x轴上的名字
# plt.ylabel("平台存活数")  # y轴上的名字
# plt.plot(x,red_y1, 'r', label='red_real')
# plt.plot(x,red_y2, 'b', label='red_next')
# plt.plot(x,red_y3, 'k', label='red_result')
# plt.plot(x,red_y4, 'g', label='red_static')
# plt.plot(x,blue_y1, 'pink', label='blue_real')
# plt.plot(x,blue_y2, 'c', label='blue_next')
# plt.plot(x,blue_y3, 'y', label='blue_result')
# plt.plot(x,blue_y4, 'm', label='blue_static')
# plt.legend(loc='best')
# # plt.savefig(os.path.join(rc().getImagePath(), self.file_name + r'.png'))
# plt.show()

# from DataOperator.mysqlOperator import mysqlOperator
# def boolToInt(input):  # 将布尔类型转化为0/1
#     res = [0] * len(input)
#     for i in range(len(input)):
#         if input[i] == True:
#             res[i] = 1
#         elif input[i] == False:
#             res[i] = 0
#         else:
#             res[i] = -1
#     return res
#
# mysqlOperator = mysqlOperator()
# init = list(mysqlOperator.FuzzyQueryAttackTagStaticData('Attack001').get('AttackInit'))
# last = list(mysqlOperator.FuzzyQueryAttackTagStaticData('Attack001').get('AttackLast'))
# svv_list = []
# for last_i in last:
#     svv_list.append(last_i['isSurvey'])
# # svv_list = last['isSurvey']
# red_list = []
# for init_i in init:
#     red_list.append(init_i['isRed'])
#
# svv_list =boolToInt(svv_list)
#
# print(svv_list)
# print(len(red_list))
#
# count = [[1], [2], [3], [4], [5], [6], [7]]
# print(count[0],count[1][-1])