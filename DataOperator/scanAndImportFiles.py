import os
import time
from threading import Thread, ThreadError

from DataOperator.mysqlOperator import mysqlOperator as mo
from DataOperator.fluxdbOperator import fluxdbOperator

# todo 使用 watchdog 模块重写代码
global killed
killed = 0


class scanner(Thread):
    def __init__(self, warPath, sleep_time=5):
        Thread.__init__(self)
        self.insert_dynamic_json_number = 0  ## 表示文件夹中的json文件数目
        self.insert_static_data = False
        self.inserted_file_list = []
        self.warPath = warPath  ## ../xxx/json_output_5055555
        # self.war_name = os.path.
        ## 拆分zz名称
        if str.endswith(self.warPath, '\\') or str.endswith(self.warPath, '/'):
            self.war_name = os.path.basename(self.warPath[:-1])
        else:
            self.war_name = os.path.basename(self.warPath)
        self.stop = 100  ## 表示线程停止时间，默认100轮，每次执行插入后都会续1秒
        self.sleep_time = sleep_time  ## 每隔5秒扫描一次指定文件夹
        # global killed = 0

    def scanDynamicData(self):
        ## warPath 为一场 zz 的路径
        dynamicDataPath = os.path.join(self.warPath, 'push')
        insert_influxdb_data = fluxdbOperator()

        if not os.listdir(dynamicDataPath):
            self.insert_dynamic_json_number = 0
            pass
        else:
            json_data_list = list(os.listdir(dynamicDataPath))
            json_num = len(json_data_list)  ## 当前文件夹中的json 文件数目

            if json_num == self.insert_dynamic_json_number:
                ## 没有新的json 文件
                pass
            else:
                new_json_name_list = []
                for i in json_data_list:
                    if i not in self.inserted_file_list:
                        new_json_name_list.append(i)
                if len(new_json_name_list) > 0:
                    for new_json_name in new_json_name_list:
                        new_json_location = os.path.join(dynamicDataPath,
                                                         new_json_name)  ## new_json_location 就是新文件的完整 location
                        ## todo 调用influxdb存储上面的新json文件
                        # print("已保存", self.war_name, "中的", new_json_name)
                        insert_influxdb_data.insert(self.war_name, new_json_location)
                        self.inserted_file_list.append(new_json_name)
                        self.insert_dynamic_json_number += 1  ## 动态数据插入成功

    def scanStaticData(self):
        ## warPath 为一场 zz 的路径
        staticDataPath = os.path.join(self.warPath, 'R')
        if self.insert_static_data:  ## 已经插入静态数据
            pass
        else:
            ## 若尚未插入静态数据，则插入
            if not os.listdir(staticDataPath):  ## 文件夹空
                assert not self.insert_static_data
                pass
            else:
                # 子文件夹不为空，则插入json文件
                static_json_name = list(os.listdir(staticDataPath))[0]
                new_json_location = os.path.join(staticDataPath,
                                                 static_json_name)  ## new_json_location 就是新文件的完整 location
                mo().insertStaticDataFromJson(war_name=self.war_name, json_location=new_json_location)
                # print("已保存", self.war_name, "中的", static_json_name)
                self.insert_static_data = True  ## 静态数据插入成功

    def run(self):
        # 把要执行的代码写到 run函数里面 线程在创建后会直接运行run函数
        while (self.insert_dynamic_json_number < 1000 and self.stop > 0):
            current_static_flag = self.insert_static_data
            current_dynamic_flag = self.insert_dynamic_json_number

            self.scanStaticData()
            self.scanDynamicData()

            if current_dynamic_flag < self.insert_dynamic_json_number or \
                    (current_static_flag is False and self.insert_static_data is True):
                self.stop += 2
            self.stop -= 1
            time.sleep(self.sleep_time)  ## 每隔5秒扫描一次文件夹
            print("self.static=", self.insert_static_data, end='\t')
            print("self.dynamic=", self.insert_dynamic_json_number, end='\t')
            print("self.stop=", self.stop)

            if self.stop % 10 == 0:
                if killed:
                    print("程序终止")
                    return

        print("已导入完毕")

    def stop(self):
        self.stop = -1
        raise ThreadError("扫描线程不活跃，未检测到需要导入的json文件，已退出")


def stopScanner():
    ## 手动停止一个线程
    global killed
    killed = 1
