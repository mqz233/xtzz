def prepareAllStaticData(self):
    warDataDirList = os.listdir(self.dataRootDirPath)  ## 全是 json_output_000001

    ## 清除数据中的非文件夹路径，xxx/xxx/..数据/
    for dir in warDataDirList:
        if not str.startswith(dir, 'json'):
            warDataDirList.remove(dir)

    allStaticPreparedData = {}
    allSvvData = {}
    ## 遍历战役文件夹，并提取所有战役中的静态数据
    for warDataDir in warDataDirList:

        currentWarStaticData = {}
        ## warDataDir 即为每场战役的文件夹名称
        staticDirPath = os.path.join(self.dataRootDirPath, warDataDir) + r'/R/'
        ## staticDataPath : xxxx/xx/R/
        jsonFileList = os.listdir(staticDirPath)

        ## 选择第一个json文件进行静态数据的样本化提取
        staticFilePath = staticDirPath + jsonFileList[0]

        if os.path.exists(staticFilePath):
            ## 获取第一个json文件数据内的全部内容，并转化为字典
            staticData = jo().convertJsonToDict(staticFilePath)
            platformList = staticData['platforms']
            ## 提取所有 platform 中 svv 字段
            svv_list = []
            for i in platformList:
                if i['svv']:
                    svv_list.append(1)
                else:
                    svv_list.append(0)
            # svv_list = [i['svv'] for i in platformList]

            for platform_i in range(len(platformList)):
                platformList[platform_i] = jo().dict_slice(platformList[platform_i], 5, 16)
                platformList[platform_i] = self.processPlatformData(platformList[platform_i])

            # staticData['platforms'] = [list(i.values()) for i in platformList]

            # currentWarStaticData[warDataDir]=list(staticData.values())
            # allStaticPreparedData[warDataDir]=list(staticData.values())

            currentWarStaticData = []
            for i in platformList:
                currentWarStaticData = currentWarStaticData + list(i.values())

            allStaticPreparedData[warDataDir] = currentWarStaticData
            allSvvData[warDataDir] = svv_list
            location = os.path.join(self.preprocessedStaticDataPath, warDataDir + '_preprocessed.json')
            jo().storeStaticProcessedData(currentWarStaticData, location)
        else:
            print("Error!")
    self.staticRawDataSet = allStaticPreparedData
    self.svvDataSet = allSvvData
    # self.staticDataSet = currentWarStaticData


def prepareAllDynamicData(self):
    warDataDirList = os.listdir(self.dataRootDirPath)  ## 全是 json_output_000001

    ## 清除数据中的非文件夹路径，xxx/xxx/..数据/
    for dir in warDataDirList:
        if not str.startswith(dir, 'json'):
            warDataDirList.remove(dir)

    allDynamicPreparedData = {}
    allSvvData = {}

    ## 遍历战役文件夹，并提取所有战役中的动态数据
    for warDataDir in warDataDirList:
        currentWarAllDynamicPreparedData = {}
        currentWarSvvDynamicPreparedData = {}

        ## warDataDir 即为每场战役的文件夹名称
        staticDirPath = os.path.join(self.dataRootDirPath, warDataDir) + r'/Push/'
        ## staticDataPath : xxxx/xx/R/
        jsonFileList = os.listdir(staticDirPath)
        ## 遍历文件夹下所有的 json 文件
        for jsonFileName in jsonFileList:
            # staticFilePath = staticDirPath + jsonFileName
            # if os.path.exists(staticFilePath):
            #     staticData = jo().convertJsonToDict(staticFilePath)
            staticFilePath = staticDirPath + jsonFileName

            if os.path.exists(staticFilePath):
                ## 获取第一个json文件数据内的全部内容，并转化为字典
                dynamicData = jo().convertJsonToDict(staticFilePath)

                ## 逐层深入，将字典转化为纯粹的数值
                contentList = dynamicData['content']['data']
                svvList = contentList['svv']
                currentJsonSVV = [0] * len(svvList)
                for i in range(len(svvList)):
                    currentJsonSVV[i] = 1 if svvList[i] else 0

                # content_Data_List = contentList['data']
                contentList = jo().dict_slice(contentList, 3, 13)
                ## 向上，将数值覆盖掉原来的字典
                content_Data_Value = list(contentList.values())
                # contentList['data'] = content_Data_Value
                currentJsonValue = []
                for i in content_Data_Value:
                    currentJsonValue = currentJsonValue + i

                currentWarAllDynamicPreparedData[jsonFileName] = currentJsonValue
                currentWarSvvDynamicPreparedData[jsonFileName] = currentJsonSVV

                warLocation = os.path.join(self.preprocessedDynamicDataPath, warDataDir)
                location = os.path.join(warLocation, jsonFileName + '_preprocessed.json')
                # jo().storeStaticProcessedData(currentWarAllDynamicPreparedData, location)
                jo().storeStaticProcessedData(currentJsonValue, location)
                print("已保存", warDataDir, "中的", jsonFileName)

                warLocation = os.path.join(self.preprocessedSvvDataPath, warDataDir)
                svvLocation = os.path.join(warLocation, jsonFileName + '_svv.json')
                # jo().storeDynamicProcessedData(currentWarSvvDynamicPreparedData,svvLocation)
                jo().storeDynamicProcessedData(currentJsonSVV, svvLocation)

        allDynamicPreparedData[warDataDir] = currentWarAllDynamicPreparedData
        allSvvData[warDataDir] = currentWarSvvDynamicPreparedData
        print("已保存", warDataDir, "\n")
    self.dynamicRawDataSet = allDynamicPreparedData
    self.svvDataSet = allSvvData


    def processPlatformData(self, adict):
        keys = list(adict.keys())
        dict_slice = {}
        # keys.remove('svv') ## 不考虑 svv

        for k in keys:
            # 如果是bool类型，则划分为 0,1
            if isinstance(adict[k], bool):
                dict_slice[k] = 1 if adict[k] else 0

            elif isinstance(adict[k], list) and isinstance(adict[k][0], bool):
                ## 若是 bool 类型的 list，则全部划分为 0,1
                dict_slice[k] = []
                for i in range(len(adict[k])):
                    # adict[k][i] = 1 if adict[k][i] else 0
                    if adict[k][i]:
                        dict_slice[k].append(1)
                    else:
                        dict_slice[k].append(0)
                # dict_slice[k] = adict[k]

            # elif isinstance(adict[k], float):
            #     dict_slice[k] = decimal.Decimal(str(adict[k]))

            elif isinstance(adict[k], str):
                if adict[k].startswith('uav'):
                    dict_slice[k] = int(adict[k][3:])

                else:
                    dict_slice[k] = adict[k]

            else:
                dict_slice[k] = adict[k]

        # assert len(dict_slice)==(end-start)
        return dict_slice


    def rawDatasetSlice(self, rawDataset: list, BATCH_SIZE: int):
        data_x = []
        data_y = []
        for i in range(len(rawDataset) - BATCH_SIZE):
            data_x.append(rawDataset[i:i + BATCH_SIZE])
            data_y.append(rawDataset[i + BATCH_SIZE])
        return np.asarray(data_x), np.asarray(data_y)


    def boolToInt(self, input: list):
        res = [0] * len(input)
        for i in range(len(input)):
            res[i] = 1 if res[i] else 0
        return res