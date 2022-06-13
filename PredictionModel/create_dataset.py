import json
import os
import numpy as np
from DataOperator.fluxdbOperator import fluxdbOperator

class Createdataset:
    def __init__(self, measurement):
        self.measurement = measurement

    def createdataset(self):

        # 某一场作战路径
        path = '../output/plane'
        output_path = './output.json'
        file_list = os.listdir(path)

        #最终x列表
        fx_list = []
        y_list = []

        #for循环遍历每一帧
        for filename in file_list:
            #打开每一帧
            with open(path+'/'+filename, 'r', encoding='utf-8') as fp:
                # 临时列表
                list = []
                # 单个x列表
                x_list = []
                #加载json数据
                json_data = json.load(fp)
                #获得时序数据
                data = json_data['plane']
                #y列表
                y_list.append(data['svv'])
                #删除非时序属性
                del data['sences']
                del data['frameId']
                del data['time']
                del data['name']
                del data['svv']
                #下面两个for循环作用是把数据摊开成一维数组
                for key in data.keys():
                    if(type(data[key]) == type([])):
                        data[key] = np.array(data[key]).flatten()
                    list.append(data[key])
                for i in range(len(list)):
                    if(type(list[i])==type(1)):
                        x_list.append(list[i])
                    else:
                        for j in range(len(list[i])):
                            x_list.append(list[i][j])
                #单个x加入到最终x
                fx_list.append(x_list)

        #形成json格式
        json_sample = {'data_x': np.array(fx_list).tolist(), 'data_y': np.array(y_list).tolist()}
        if not os.path.exists(os.path.dirname(output_path)):
                os.mkdir(os.path.dirname(output_path))

        with open(output_path, 'w', encoding='utf-8') as fp:
                json.dump(json_sample, fp)

        # print(fx_list,y_list)
        # print("-")
        return output_path

    def createdataset2(self):

        result = fluxdbOperator().select_num_battle(self.measurement)

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

        return fxlist,ylist

    def red_blue(self):
        result = fluxdbOperator().select_num_battle(self.measurement)

        list = json.loads(result[0]['isRed'])

        return list



