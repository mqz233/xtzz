import os

import numpy as np

from DataOperator.fluxdbOperator import fluxdbOperator as fo
from DataOperator.mysqlOperator import mysqlOperator as mo
from conf.readConfig import readConfig as rc
from jsonOperator import jsonOperator as jo
from DataOperator.read_json import Read_json
import json

class createDataSet:
    def __init__(self):
        self.dataRootDirPath = rc().getDataRootDirPath()  ## 根目录
        # self.war_number = len(os.listdir(self.dataRootDirPath))  ##
        self.staticRawDataSet = {}  ## 键值对，key 为作战文件夹名称，value 为对应的静态数据值
        self.svvDataSet = {}  ## 键值对，key 为作战文件夹名称，value 为对应的存活数据
        self.dynamicRawDataSet = {}  ## 键值对，key 为作战文件夹名称，value 为一个列表，列表中的每一个元素都是一个Json导出的动态数据值
        # self.preprocessedStaticDataPath = rc().getPreprocessedStaticDataPath()  ## 处理后的静态、动态、目标数据
        # self.preprocessedDynamicDataPath = rc().getPreprocessedDynamicDataPath()
        # self.preprocessedSvvDataPath = rc().getPreprocessedSvvDataPath()
        # self.preprocessedSamplesDataPath = rc().getPreprocessedSamplesDataPath()
        # self.datasetPath = rc().getDatasetPath()  ## 数据集位置
        # self.staticDatasetPath = rc().getStaticDatasetPath()  ## 全静态数据集位置

    def boolToInt(self, input):#将布尔类型转化为0/1
        res = None
        if type(input)=='bool':
            res = 1 if input else 0
        elif type(input)=='list':
            res = [0] * len(input)
            for i in range(len(input)):
                res[i] = 1 if input[i] else 0
        else:
            pass
        return res

    def unfoldList(self,alist):#把comm多维数组铺开
        return list(np.array(alist).reshape(-1))

    def processDictFeaturesForDataset(self,adict,type:str):
        """
        对接受到的平台字典进行处理，筛掉其中的某些属性，并把所有高维列表进行展开
        :param adict: 待处理的字典
        :param type: 表示静态、动态字典
        :return: 返回处理后的新字典
        """
        if type == 'pure_static':
            adict.pop('uav_id')
            adict.pop('svv')
            adict.pop('type')
            adict['master'] = self.boolToInt(adict['master'])
            adict['radar'] = self.boolToInt(adict['radar'])

        elif type == 'static':
            adict.pop('uav_id')
            adict.pop('svv')
            adict.pop('type')
            adict.pop('master')
            adict.pop('radar')
            adict.pop('posx')
            adict.pop('posy')
            adict.pop('posz')
            adict.pop('v')
            adict.pop('northv')
            adict.pop('eastv')
            adict.pop('upv')
            adict.pop('psi')
            adict.pop('gv')
            adict.pop('iv')

        elif type == 'dynamic':
            adict['svv'] = self.boolToInt(adict['svv'])
            # adict['master'] = self.boolToInt(adict['master'])
            adict['radar'] = self.boolToInt(adict['radar'])
            adict['comm'] = self.unfoldList(adict['comm'])

        else:
            print("type error!")
        return adict


    def get_file(self, dir_path):
        list_json = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                list_json.append(os.path.join(root, file))
        return list_json

    def get_static_platform_data_list(self):
        pass
    def get_dynamic_data_list(self,dynamicDirPath):
        list_json = self.get_file(dynamicDirPath)
        dynamic_data_list = []
        for json_path in list_json:
            if json_path.find('R') >= 0:
                continue
            with open(json_path, 'rb') as f:
                params = json.load(f)
            data = Read_json(params)
            data.data_analysis()
            dynamic_data_list += list(data.posx)+list(data.posy)+list(data.posz)+list(data.v)+list(data.northv)+list(data.upv)+list(data.eastv)+list(data.psi)+list(data.gv)+list(data.iv)
            # ["posx","posy","posz","v","northv","upv","eastv","psi","gv","iv"]
        return dynamic_data_list
    # def create_realtime_DataSet(self,staticDirPath,dynamicDirPath):
    def create_realtime_DataSet(self,static_platform_data,dynamic_data_list):
        #处理动态数据
        # list_json = self.get_file(dynamicDirPath)
        # result_json = []
        # for json_path in list_json:
        #     if json_path.find('R') >= 0:
        #         continue
        #     with open(json_path, 'rb') as f:
        #         params = json.load(f)
        #     data = Read_json(params)
        #     data.data_analysis()
        # ## 先获取本场 zz的数据数值，无名称，一维list结构，
        # # 静态 len = 10 * 属性， 动态为 二维列表，1000*属性值
        # static_war_data,static_platform_data = self.obtainStaticDataFromDatabase(dirName)
        # dynamic_data_list = self.obtainDynamicDataFromDatabase(dirName) ## 获取本场zz中的所有动态数据，包括svv

        target = 'svv'

        ## 纯静态预测数据集：
        single_pure_static_dataset = []
        for i in static_platform_data:
            i = self.processDictFeaturesForDataset(i, type='pure_static')
            current_pure_static_values = list(i.values())
            single_pure_static_dataset = single_pure_static_dataset + current_pure_static_values



        ## 动态加静态预测，静态部分：
        current_war_static_dataset = []

        for i in static_platform_data:
            i = self.processDictFeaturesForDataset(i,type = 'static')
            current_platform_values = list(i.values())
            current_war_static_dataset= current_war_static_dataset + current_platform_values


        current_war_target_dataset = []  ## len target = len x
        current_war_dynamic_dataset = []

        # 遍历当前zz中所有的 json（时刻），并取出动态、svv数据，建立混合数据集#
        for current_frame_data in dynamic_data_list:
            ## svv
            current_war_target_dataset.append(current_frame_data[target])

            ## dynamic
            current_frame_data = self.processDictFeaturesForDataset(current_frame_data,'dynamic')
            current_frame_dynamic_values = list(current_frame_data.values())
            current_frame_static_plus_dynamic = current_war_static_dataset + current_frame_dynamic_values
            current_war_dynamic_dataset.append(current_frame_static_plus_dynamic)

        ## 纯静态扩大len（svv）倍
        pure_static_dataset = []
        for i in range(len(current_war_target_dataset)):
            pure_static_dataset = pure_static_dataset + single_pure_static_dataset


        ## todo:组合这几个量

        return pure_static_dataset,current_war_dynamic_dataset,current_war_target_dataset
        #todo:list转化为json


    def obtainStaticDataFromDatabase(self, war_name):
        ## 获取本场zz中除svv外的其他数据
        static_features = mo().getValuableStaticFeatures()
        static_features.remove('svv')

        queryList = [war_name] + static_features
        staticData = mo().queryStaticData(queryList)
        staticWarData,staticPlatformsData = list(staticData['war_data']),list(staticData['platform_data'])
        return staticWarData,staticPlatformsData

    def obtainDynamicDataFromDatabase(self, war_name):
        ## 获取本场zz中的所有动态数据，包括svv
        dynamicData = fo().select_num_battle(measurement=war_name)
        return dynamicData

    def createDataSet(self):
        ## 生成混合数据集 与 静态数据集， 其中所有属性均有截取
        # staticDataPath = self.preprocessedStaticDataPath
        # dynamicDataPath = self.preprocessedDynamicDataPath
        # svvDataPath = self.preprocessedSvvDataPath
        # datasetPath = self.datasetPath

        ## 全静态数据集： 静态 + 静态
        ## 混合数据集：   静态 + 动态 （每场作战的每个 slot）

        ## 查找zz场数
        warDataDirList = mo().queryWarData()  ##  json_output_000001 文件夹的集合

        ## 将从文件中获取数据修改为从数据库中获取数据
        ## 数据准备，将数据存入数据库
        ## todo 添加多进程调度，将扫描文件加入此处

        ## 遍历每一场 zz 文件夹
        for dir_index in range(len(warDataDirList)):
            dirName = warDataDirList[dir_index]

            ## 先获取本场 zz的数据数值，无名称，一维list结构，
            # 静态 len = 10 * 属性， 动态为 二维列表，1000*属性值
            static_war_data,static_platform_data = self.obtainStaticDataFromDatabase(dirName)
            dynamic_data_list = self.obtainDynamicDataFromDatabase(dirName)

            target = 'svv'

            ## 纯静态预测数据集：
            single_pure_static_dataset = []
            for i in static_platform_data:
                i = self.processDictFeaturesForDataset(i, type='pure_static')
                current_pure_static_values = list(i.values())
                single_pure_static_dataset = single_pure_static_dataset + current_pure_static_values



            ## 动态加静态预测，静态部分：
            current_war_static_dataset = []

            for i in static_platform_data:
                i = self.processDictFeaturesForDataset(i,type = 'static')
                current_platform_values = list(i.values())
                current_war_static_dataset= current_war_static_dataset + current_platform_values


            current_war_target_dataset = []  ## len target = len x
            current_war_dynamic_dataset = []

            # 遍历当前zz中所有的 json（时刻），并取出动态、svv数据，建立混合数据集#
            for current_frame_data in dynamic_data_list:
                ## svv
                current_war_target_dataset.append(current_frame_data[target])

                ## dynamic
                current_frame_data = self.processDictFeaturesForDataset(current_frame_data,'dynamic')
                current_frame_dynamic_values = list(current_frame_data.values())
                current_frame_static_plus_dynamic = current_war_static_dataset + current_frame_dynamic_values
                current_war_dynamic_dataset.append(current_frame_static_plus_dynamic)

            ## 纯静态扩大len（svv）倍
            pure_static_dataset = []
            for i in range(len(current_war_target_dataset)):
                pure_static_dataset = pure_static_dataset + single_pure_static_dataset


            ## todo:组合这几个量

            return pure_static_dataset,current_war_dynamic_dataset,current_war_target_dataset

    def createDataSet2(self):
        ## 生成混合数据集 与 静态数据集， 其中所有属性均有截取
        staticDataPath = self.preprocessedStaticDataPath
        dynamicDataPath = self.preprocessedDynamicDataPath
        svvDataPath = self.preprocessedSvvDataPath
        datasetPath = self.datasetPath

        ## 全静态数据集： 静态 + 静态
        ## 混合数据集：   静态 + 动态 （每场作战的每个 slot）

        ## 查找zz场数
        warDataDirList = os.listdir(dynamicDataPath)  ##  json_output_000001 文件夹的集合
        staticFileList = os.listdir(staticDataPath)  ## zz name 文件

        ## 将从文件中获取数据修改为从数据库中获取数据
        ## 准备
        if len(warDataDirList) < self.war_number:
            self.prepareAllDynamicData()
        if len(staticFileList) < self.war_number:
            self.prepareAllStaticData()

        assert len(warDataDirList) == len(staticFileList)  ## 动、静态 zz 场数相同

        ## 遍历每一场 zz 文件夹
        for dir_index in range(len(warDataDirList)):
            dirName = warDataDirList[dir_index]

            ## 先获取本场 zz的静态数据
            staticDataIndex = staticFileList.index(dirName + '_preprocessed.json')
            staticData = jo().convertJsonToList(os.path.join(staticDataPath, staticFileList[staticDataIndex]))

            ## 获取每一场作战的所有动态数据，每一个值都代表一个 Slot 下动态数据的取值
            currentJsonDynamicPath = os.path.join(dynamicDataPath, dirName)
            currentJsonSvvPath = os.path.join(svvDataPath, dirName)

            currentDynamicJsonList = os.listdir(currentJsonDynamicPath)  ## 获取所有json文件名
            currentSvvJsonList = os.listdir(currentJsonSvvPath)  ## 获取所有json文件名

            currentWarTrainData = []
            currentWarTestData = []

            ## 遍历当前zz中所有的 json（时刻），并取出动态、svv数据，建立混合数据集
            for dynamicJsonName in currentDynamicJsonList:
                currentJsonDynamicData = jo().convertJsonToList(os.path.join(currentJsonDynamicPath, dynamicJsonName))
                currentJsonTrainData = staticData + currentJsonDynamicData
                currentWarTrainData.append(currentJsonTrainData)

            for svvJsonName in currentSvvJsonList:
                currentJsonSvvData = jo().convertJsonToList(os.path.join(currentJsonSvvPath, svvJsonName))
                currentJsonTestData = currentJsonSvvData
                currentWarTestData.append(currentJsonTestData)

            ## 断言：原始数据集中，x 变量个数 与 y 变量个数相同
            assert len(currentWarTrainData) == len(currentWarTestData)

            ## 创建当前 zz 中的混合数据集, x: 属性值，y：属性值下一时刻的真实值，z：属性值对应的 Svv
            # current_war_train_x,current_war_train_y = self.rawDatasetSlice(currentWarTrainData,BATCH_SIZE)
            # current_war_test_z = self.rawDatasetSlice(currentWarTestData,BATCH_SIZE)
            # current_war_Sample = {'data_x':current_war_train_x,'data_y':current_war_train_y,'data_z':current_war_test_z}
            current_war_Sample = {'data_x': currentWarTrainData, 'data_y': currentWarTestData}

            static_train_sample = []
            for i in range(len(currentWarTestData)):
                static_train_sample.append(staticData)
            # static_sample_x, static_sample_y= list(self.rawDatasetSlice(static_train_sample,BATCH_SIZE))
            current_war_static_sample = {'data_x': static_train_sample, 'data_y': currentWarTestData}
            jo().storeStaticProcessedData(current_war_Sample,
                                          location=os.path.join(datasetPath, dirName + '_dynamic.json'))
            jo().storeStaticProcessedData(current_war_static_sample,
                                          location=os.path.join(datasetPath, dirName + '_static.json'))
            print("已保存", dirName, "生成的样本数据")

    def createFullSlotDataset(self):
        ## todo 对所有json都创建输入，不够处填0
        pass




# createDataSet().createDataSet()
# createDataSet().prepareAllDynamicData()
