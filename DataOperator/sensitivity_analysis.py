import json

from DataOperator.fluxdbOperator import fluxdbOperator
from DataOperator.influxdb_data_store import Dynamic
from DataOperator.mysqlOperator import mysqlOperator

class sensitivity_analysis():
    def __init__(self) -> None:
        # self.root_path = self.get_path()
        self.init_value = [300, 60, 0, 0, 150, 80, 0, 0, 220, 80, 0, 0, 220, 80, 0, 0]

        self.col_range = []
        # self.count = []

        type_step = [10.0, 5.0, 1, 30, 10.0, 5.0]
        type_min = [40.0, 10.0, 1, 30, 70.0, 10.0]

        for i in range(6):
        ## 共 6 符图
            for x in range(10):
                x_label = [type_min[i] + x * type_step[i]]
                self.col_range.append(x_label)
        assert 1==1

        ## 40 个平台
        """
        1212 1209、1210 1207、1208 1206 1204 1211 1205
        1208 : 4 个  0 ~ 3
        1207 : 8 个  4 ~ 11
        1204 : 1 个  12
        1205 - 2     13 ~ 14
        1211 - 4     15 ~ 18
        1212 - 4     19 ~ 22
        1209 : 6 个  23 ~ 24 26-27 29-30
        1210 : 3 个  25 28 31
        1206 : 8 个  32 ~ 39
        """
    # def __init__(self) -> None:
    #     # self.root_path = self.get_path()
    #     self.init_value = [300, 60, 0, 0, 150, 80, 0, 0, 220, 80, 0, 0, 220, 80, 0, 0]
    #
    #     self.col_range = []
    #     # self.count = []
    #
    #     type_step = [20.0, 5, 10.0, 0.5, 10.0, 5, 10.0, 0.5, 30.0, 5, 10.0, 0.5, 20.0, 5, 10.0, 0.5]
    #     type_min = [120.0, 15, 50.0, 0.5, 80.0, 10, 50.0, 0.5, 100.0, 15, 50.0, 0.5, 20.0, 5, 50.0, 0.5]
    #
    #     for i in range(12):
    #     ## 共 12 符图
    #         x_label = [type_min[i] + x * type_step[i] for x in range(10)]
    #         self.col_range.append(x_label)


        ## 40 个平台
        """
        1212 1209、1210 1207、1208 1206 1204 1211 1205
        1212 : 4 个  0 ~ 3
        1209 : 6 个  4 ~ 9
        1208 : 4 个  10 ~ 13
        1207 - 8     14 ~ 21
        1206 - 8     22 ~ 29
        1204 - 1     30
        1211 : 4 个  31 ~ 34
        1210 : 3 个  35 ~ 37
        1205 : 2 个  38 ~ 39
        """
    def boolToInt(self, input):#将布尔类型转化为0/1
        res = [0] * len(input)
        for i in range(len(input)):
            input[i]=input[i].strip()
            if input[i] == 'True':
                res[i]=1
            elif input[i] == 'False':
                res[i]=0
            else:
                res[i] =-1
        return res

    def boolToInt2(self, input):#将布尔类型转化为0/1
        res = [0] * len(input)
        for i in range(len(input)):
            if input[i] == True:  #注意是字符串还是bool型
                res[i]=1
            elif input[i] == False:
                res[i]=0
            else:
                res[i] =-1
        return res

    def get_x_label(self, col_index):
        """
        获取当前属性的所有取值范围，作为绘图的X轴
        """
        # return self.col_range[col_index-1]
        x_label= self.col_range[10*(col_index-1):10*(col_index-1)+9]
        return [i[0] for i in x_label]

    def get_graph_data(self, col_index, tag):
        graph_index = col_index - 1  # col_index = [1-12]

        # todo 输入作战场数，获取动态数据表中最后的svv属性
        client = fluxdbOperator()
        measurements = client.get_plane_measurements()
        table = [] # 需要统计的表
        for m in measurements:
            if tag in m:
                table.append(m)
        ## 循环获取每一场战斗的结果
        count = [[], [], [], [], [], [], []]  ## 所有飞机、全部作战结果
        cnt = 0
        for war in table:
            result = client.select_num_battle(war)[-1]
            ## 获取战斗结果
            svv = json.loads(result['svv'])
            # 型号
            type = json.loads(result['type'])
            #红蓝方
            isred = json.loads(result['isRed'])

            ########################### 1212 1209、1210 1207、1208 1206 1204 1211 1205
            count[0].append(0)
            count[1].append(0)
            count[2].append(0)
            count[3].append(0)
            count[4].append(0)
            count[5].append(0)
            count[6].append(0)
            for i in range(len(svv)):
                if type[i] == 1212:
                    count[0][cnt] += svv[i]
                elif type[i] == 1209 or type[i] == 1210:
                    count[1][cnt] += svv[i]
                elif type[i] == 1207 or type[i] == 1208:
                    count[2][cnt] += svv[i]
                elif type[i] == 1206:
                    count[3][cnt] += svv[i]
                elif type[i] == 1204:
                    count[4][cnt] += svv[i]
                elif type[i] == 1211:
                    count[5][cnt] += svv[i]
                elif type[i] == 1205:
                    count[6][cnt] += svv[i]
            cnt += 1

        return {'x': self.get_x_label(col_index), 'y': count}

    def get_graph_data_red(self, col_index, tag):
        graph_index = col_index - 1  # col_index = [1-12]

        # 输入作战场数，获取动态数据表中最后的svv属性
        client = fluxdbOperator()
        measurements = client.get_plane_measurements()
        table = [] # 需要统计的表
        for m in measurements:
            if tag in m:
                table.append(m)
        ## 循环获取每一场战斗的结果
        count = [[], [], [], [], [], [], []]  ## 所有飞机、全部作战结果
        cnt = 0
        for war in table:
            result = client.select_num_battle(war)[-1]
            ## 获取战斗结果
            svv = json.loads(result['svv'])
            # 型号
            type = json.loads(result['type'])
            #红蓝方
            isred = json.loads(result['isRed'])

            ########################### 1212 1209、1210 1207、1208 1206 1204 1211 1205
            count[0].append(0)
            count[1].append(0)
            count[2].append(0)
            count[3].append(0)
            count[4].append(0)
            count[5].append(0)
            count[6].append(0)
            for i in range(len(svv)):
                if isred[i] == 1:
                    if type[i] == 1212:
                        count[0][cnt] += svv[i]
                    elif type[i] == 1209 or type[i] == 1210:
                        count[1][cnt] += svv[i]
                    elif type[i] == 1207 or type[i] == 1208:
                        count[2][cnt] += svv[i]
                    elif type[i] == 1206:
                        count[3][cnt] += svv[i]
                    elif type[i] == 1204:
                        count[4][cnt] += svv[i]
                    elif type[i] == 1211:
                        count[5][cnt] += svv[i]
                    elif type[i] == 1205:
                        count[6][cnt] += svv[i]
            cnt += 1

        return {'x': self.get_x_label(col_index), 'y': count}

    def get_graph_data_blue(self, col_index, tag):
        graph_index = col_index - 1  # col_index = [1-12]

        # 输入作战场数，获取动态数据表中最后的svv属性
        client = fluxdbOperator()
        measurements = client.get_plane_measurements()
        table = [] # 需要统计的表
        for m in measurements:
            if tag in m:
                table.append(m)
        ## 循环获取每一场战斗的结果
        count = [[], [], [], [], [], [], []]  ## 所有飞机、全部作战结果
        cnt = 0
        for war in table:
            result = client.select_num_battle(war)[-1]
            ## 获取战斗结果
            svv = json.loads(result['svv'])
            # 型号
            type = json.loads(result['type'])
            #红蓝方
            isred = json.loads(result['isRed'])

            ########################### 1212 1209、1210 1207、1208 1206 1204 1211 1205
            count[0].append(0)
            count[1].append(0)
            count[2].append(0)
            count[3].append(0)
            count[4].append(0)
            count[5].append(0)
            count[6].append(0)
            for i in range(len(svv)):
                if isred[i] != 1:
                    if type[i] == 1212:
                        count[0][cnt] += svv[i]
                    elif type[i] == 1209 or type[i] == 1210:
                        count[1][cnt] += svv[i]
                    elif type[i] == 1207 or type[i] == 1208:
                        count[2][cnt] += svv[i]
                    elif type[i] == 1206:
                        count[3][cnt] += svv[i]
                    elif type[i] == 1204:
                        count[4][cnt] += svv[i]
                    elif type[i] == 1211:
                        count[5][cnt] += svv[i]
                    elif type[i] == 1205:
                        count[6][cnt] += svv[i]
            cnt += 1

        return {'x': self.get_x_label(col_index), 'y': count}

    def get_graph_data2(self, col_index,tag):
        graph_index = col_index - 1  # col_index = [1-6]

        # todo 输入作战场数，获取动态数据表中最后的svv属性
        war_start_id = graph_index * 10 + 1
        war_end_id = col_index * 10 + 1

        ## 循环获取每一场战斗的结果
        count = [[], [], [], [], [], [], []]  ## 所有飞机、全部作战结果
        for war_id in range(war_start_id, war_end_id):
            war_name = tag
            if war_id < 10:
                war_name = war_name + '00' + str(war_id)

            elif war_id >= 10 and war_id < 100:
                war_name = war_name + '0' + str(war_id)

            else:
                war_name += str(war_id)
            current_count = [0] * 7  ## 当前作战结果统计
            last = list(mysqlOperator().QueryAttackTagStaticData(war_name).get('AttackLast'))
            svv_list = []
            for last_i in last:
                svv_list.append(last_i['isSurvey'])
            # svv_list = last['isSurvey']
            svv_list = self.boolToInt2(svv_list)

            """
            1212 1209、1210 1207、1208 1206 1204 1211 1205
                 1208 : 4 个  0 ~ 3 
                 1207 : 8 个  4 ~ 11
                 1204 : 1 个  12 
                 1205 - 2     13 ~ 14
                 1211 - 4     15 ~ 18
                 1212 - 4     19 ~ 22
                 1209 : 6 个  23 ~ 24 26-27 29-30
                 1210 : 3 个  25 28 31  
                 1206 : 8 个  32 ~ 39
                 """
            # svv = [1] * 40
            # ssvv  = sum(svv)
            # assert len(svv)==40
            count[0].append(sum(svv_list[19:23]))
            count[1].append(sum(svv_list[23:25])+sum(svv_list[26:28])+sum(svv_list[29:31])+svv_list[25]+svv_list[28]+svv_list[31])
            count[2].append(sum(svv_list[4:12])+sum(svv_list[0:4]))
            count[3].append(sum(svv_list[32:40]))
            count[4].append(svv_list[12])
            count[5].append(sum(svv_list[15:19]))
            count[6].append(sum(svv_list[13:15]))

            # count.append(current_count)
        # count += 1  # 统计结果

        return {'x': self.get_x_label(col_index), 'y': count}

#
# s = sensitivity_analysis().get_graph_data2(2,'Attack')
# print(s)
# assert 1==1