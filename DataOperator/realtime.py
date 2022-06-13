from multiprocessing import Process
import time
from PredictionModel.figs import pre_result
from DataOperator.createDataSet import createDataSet
import os

class Myprocess(Process):
    def __init__(self, warPath):
        super().__init__()
        self.warPath = warPath
        self.dynamicDataPath = os.path.join(self.warPath, 'push')
        self.staticDataPath = os.path.join(self.warPath, 'R')
        self.Data_Processed = os.path.join(self.warPath, 'data_process')
        self.frame_num = len(list(os.listdir(path=self.dynamicDataPath)))

    def run(self):  # 一定要叫这个名字，不能是别的
        frame_num = len(list(os.listdir(path=self.dynamicDataPath))) #扫描动态文件下有几帧数据
        # y1,y2,y3 = pre_result(self.Data_Processed,frame_num)
        y1, y2, y3 = pre_result('..\data\json_output_5055555_dynamic.json', 5)
        return y1,y2,y3

if __name__ == '__main__':

    while(1):
        p = Myprocess('..\data')
        p.start()
        p.join()#等待子进程执行结束
    # todo:加入扫描文件，数据预处理步骤
        now_frame_num = len(list(os.listdir(path=p.dynamicDataPath)))
        if p.frame_num < now_frame_num:

            print('finish')