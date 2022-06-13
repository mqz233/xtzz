import json
import os
# import torch.

class jsonOperator:

    # def __init__(self):
    #     self.rf = readConfig()

    def convertJsonToDict(self, location):
        """
        将json 文件导入，并转化为字典格式
        :param location: json文件位置
        :return: dict 转化后的字典
        """
        # json_data = {}
        with open(location, 'r', encoding='utf-8') as fp:
            json_data = json.load(fp)
        return json_data

    def convertJsonToList(self, location) -> list:
        """
        将json 文件导入，并转化为字典格式
        :param location: json文件位置
        :return: list 转化后的列表
        """
        # json_data = {}
        with open(location, 'r', encoding='utf-8') as fp:
            json_data = json.load(fp)
        return json_data

    def storeStaticProcessedData(self, adict, location,encoding = 'gbk'):

        if not os.path.exists(os.path.dirname(location)):
            os.mkdir(os.path.dirname(location))

        with open(location, 'w', encoding=encoding) as fp:
            json.dump(adict, fp)
        return True

    def storeDynamicProcessedData(self, adict, location):
        if not os.path.exists(os.path.dirname(os.path.dirname(location))):
            os.mkdir(os.path.dirname(os.path.dirname(location)))
        if not os.path.exists(os.path.dirname(location)):
            os.mkdir(os.path.dirname(location))
        with open(location, 'w', encoding='utf-8') as fp:
            json.dump(adict, fp)
        return True

    # def getCurrentDataFolderName(self):
    #     return self.rf.getDataRootDir()

    def dict_slice(self, adict, start, end):
        """
        :param adict: 需要分片的字典
        :param start: 字典起始位置，从0开始
        :param end: 字典结束位置，到 len(adict.keys())-1 为止
        :return: 返回分片后的字典
        """
        keys = adict.keys()
        dict_slice = {}
        for k in list(keys)[start:end]:
            dict_slice[k] = adict[k]
        return dict_slice

#
# jo = jsonOperator()
# dict=jo.convertJsonToDict("000000.json")
# print(dict.keys())
# print(dict.keys())
# print(dict.items())
# # print(jsonOperator().getCurrentDataFolderName())
# print(dict["platforms"][0])
# b={'a': 'wo', 'b': 'zai', 'c': 'zhe', 'd': ['li',[[51,65],[163],[521]]]}
# location = rc().getPreprocessedDataPath()
# location+='a.json'
# jo.converDictToJson(b,location)
# b1 = jo.convertJsonToDict(location)
# print(b==b1)
