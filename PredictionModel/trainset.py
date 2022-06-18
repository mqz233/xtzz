import json
import os
import numpy as np
from DataOperator.fluxdbOperator import fluxdbOperator

class Createtrainset:
    def __init__(self,tag):
        self.client = fluxdbOperator()
        self.tag=tag
        self.stage = { ###############
            'A':0,
            'B':1,
            'C':2,
            'D':3
        }
        self.eval = { ###################
            '1': 0,
            '2':1,
            '3':2
        }

    def createtrainset(self):
        client = self.client
        measurements = client.get_plane_measurements()
        table = []  # 需要统计的表
        for m in measurements:
            if self.tag in m:
                table.append(m)
        num = int(len(table) * 0.8)
        trainset = []
        for i in range(num):
            tset = {}
            result = client.select_num_battle(table[i])
            fxlist = []
            ylist = []
            for i in range(len(result)):
                list = []
                xlist = []
                data = result[i]
                ylist.append(json.loads(data['svv']))

                del data['sences']
                del data['frameId']
                del data['Time']
                del data['time']
                del data['name']
                del data['svv']
                if 'stage' in data: #################
                    del data['stage']
                    del data['eval']

                del data['radarList']
                del data['locked']
                # del data['det_pro']
                # del data['range_acc']
                # del data['angle_acc']
                del data['atkList']
                del data['conList']
                del data['comm']
                del data['suppressList']
                del data['echo']
                del data['isRed']
                del data['type']
                # del data['value']
                # del data['ra_Pro_Angle']
                # del data['ra_Pro_Radius']
                # del data['ra_StartUp_Delay']
                # del data['ra_Detect_Delay']
                # del data['ra_Process_Delay']
                # del data['ra_FindTar_Delay']
                # del data['ra_Rang_Accuracy']
                # del data['ra_Angle_Accuracy']
                # del data['MisMaxAngle']
                # del data['MisMaxRange']
                # del data['MisMinDisescapeDis']
                # del data['MisMaxDisescapeDis']
                # del data['MisMaxV']
                # del data['MisMaxOver']
                # del data['MisLockTime']
                # del data['MisHitPro']
                # del data['MisMinAtkDis']
                # del data['MisNum']
                # del data['EchoInitState']
                # del data['EchoFackTarNum']
                # del data['EchoDis']
                # del data['SupInitState']
                # del data['SupTarNum']
                # del data['SupMinDis']
                # del data['SupMaxAngle']
                del data['comNum']
                del data['suppressNum']
                del data['echoNum']

                xlist.append(self.stage[data['stage']]) #########################
                xlist.append(self.eval[data['eval']])

                for key in data.keys(): ################################
                    data[key] = json.loads(data[key])
                    data[key] = np.array(data[key]).flatten().tolist()
                    list.append(data[key])
                for j in range(len(list)):
                    for k in range(len(list[j])):
                        xlist.append(list[j][k])
                fxlist.append(xlist)
            tset['data_x'] = fxlist
            tset['data_y'] = ylist
            trainset.append(tset)
        return trainset

    def createtrainsetno(self):
        client = self.client
        measurements = client.get_plane_measurements()
        table = []  # 需要统计的表
        for m in measurements:
            if self.tag in m:
                table.append(m)
        num = int(len(table) * 0.8)
        trainset = []
        for i in range(num):
            tset = {}
            result = client.select_num_battle(table[i])
            fxlist = []
            ylist = []
            for i in range(len(result)):
                list = []
                xlist = []
                data = result[i]
                ylist.append(json.loads(data['svv']))

                del data['sences']
                del data['frameId']
                del data['Time']
                del data['time']
                del data['name']
                del data['svv']
                if 'stage' in data:
                    del data['stage']
                    del data['eval']

                del data['radarList']
                del data['locked']
                # del data['det_pro']
                # del data['range_acc']
                # del data['angle_acc']
                del data['atkList']
                del data['conList']
                del data['comm']
                del data['suppressList']
                del data['echo']
                del data['isRed']
                del data['type']
                # del data['value']
                # del data['ra_Pro_Angle']
                # del data['ra_Pro_Radius']
                # del data['ra_StartUp_Delay']
                # del data['ra_Detect_Delay']
                # del data['ra_Process_Delay']
                # del data['ra_FindTar_Delay']
                # del data['ra_Rang_Accuracy']
                # del data['ra_Angle_Accuracy']
                # del data['MisMaxAngle']
                # del data['MisMaxRange']
                # del data['MisMinDisescapeDis']
                # del data['MisMaxDisescapeDis']
                # del data['MisMaxV']
                # del data['MisMaxOver']
                # del data['MisLockTime']
                # del data['MisHitPro']
                # del data['MisMinAtkDis']
                # del data['MisNum']
                # del data['EchoInitState']
                # del data['EchoFackTarNum']
                # del data['EchoDis']
                # del data['SupInitState']
                # del data['SupTarNum']
                # del data['SupMinDis']
                # del data['SupMaxAngle']
                del data['comNum']
                del data['suppressNum']
                del data['echoNum']

                for key in data.keys():
                    data[key] = json.loads(data[key])
                    data[key] = np.array(data[key]).flatten().tolist()
                    list.append(data[key])
                for j in range(len(list)):
                    for k in range(len(list[j])):
                        xlist.append(list[j][k])
                fxlist.append(xlist)
            tset['data_x'] = fxlist
            tset['data_y'] = ylist
            trainset.append(tset)
        return trainset