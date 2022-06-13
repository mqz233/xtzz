import os
import time
from threading import Thread, ThreadError

from DataOperator.mysqlOperator import mysqlOperator as mo
from DataOperator.fluxdbOperator import fluxdbOperator

# todo 使用 watchdog 模块重写代码
global killed
killed = 0

class threadtest(Thread):
    def __init__(self, warPath, sleep_time=5):
        Thread.__init__(self)

        self.stop = 100  ## 表示线程停止时间，默认100轮，每次执行插入后都会续1秒
        self.sleep_time = sleep_time  ## 每隔5秒扫描一次指定文件夹

    def run(self):
        print("已导入完毕")

    def stop(self):
            self.stop = -1
            raise ThreadError("线程结束")

def stopthreadtest():
        ## 手动停止一个线程
        global killed
        killed = 1