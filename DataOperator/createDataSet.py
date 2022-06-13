import os

import numpy as np
from PredictionModel.create_dataset import Createdataset
from DataOperator.fluxdbOperator import fluxdbOperator as fo
from DataOperator.mysqlOperator import mysqlOperator as mo
from conf.readConfig import readConfig as rc
# from jsonOperator import jsonOperator as jo
from DataOperator.read_json import Read_json
import json

class Nstr:
    def __init__(self, arg):
       self.x=arg
    def __sub__(self,other):
        c=self.x.replace(other.x,"")
        return c

class Warwinner:
    def __init__(self,tag):
        self.tag = tag

    def winner(self):
        client = fo()
        measurements = client.get_plane_measurements()
        table = []  # 需要统计的表
        warname = []
        for m in measurements:
            if self.tag in m:
                table.append(m)
                warname.append(m.replace('plane'+self.tag, ''))
        # 总场次
        total = len(table)
        # 红方胜场
        red = []
        #蓝方胜场
        blue = []
        #平局
        draw = []
        for i in range(total):
            dataset = Createdataset(table[i])
            xlist, ylist = dataset.createdataset2()
            isred = dataset.red_blue()
            r = 0
            b = 0
            for j in range(len(isred)):
                if isred[j] == 1:
                    r += ylist[-1][j]
                else:
                    b += ylist[-1][j]
            if r > b:
                red.append(warname[i])
            elif r < b:
                blue.append(warname[i])
            else:
                draw.append(warname[i])

        return warname, red, blue, draw

