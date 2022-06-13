import json

from influxdb import InfluxDBClient
from conf.readConfig import readConfig
from DataOperator.read_json import Read_json


class fluxdbOperator:
    def __init__(self):
        # 连接数据库服务器
        self.client = InfluxDBClient(str(readConfig().getInfluxdbHost()), int(readConfig().getInfluxdbPort()), 'root', 'root', str(readConfig().getInfluxdbBb()))

    def select_time(self, measurement: str, time):  # todo:time属性必须严格安装时间格式来存
        # sql = r"-- select COLUMN_NAME from information_schema.COLUMNS where table_name =" + "\"" + tableName + "\"" + " order by colorder ;"
        #依据query()方法写入查询语句
        result = self.client.query(
            'select * ' + 'from' + ' ' + measurement[0] + ' ' + 'where data <= ' + str(time) + ';')  # time: 2019-10-18
        return result

    def select_num_battle(self, measurement: str):  # todo:time属性必须严格安装时间格式来存
        ## 返回表（一场zz）中的所有数据
        # sql = r"-- select COLUMN_NAME from information_schema.COLUMNS where table_name =" + "\"" + tableName + "\"" + " order by colorder ;"

        war_name = measurement[0] if type(measurement)==list else measurement
        # result = self.client.query('select comm, eastv, northv, posx, posy, psi, radar, SuppressStatus, m_EchoStatus, svv, v, AttackDis, m_TargetNum, m_TargetPdList, m_TargetAccList, m_AngleAccList, m_AttackNum, m_ControlNum, m_GuideID ' + 'from' + ' ' + war_name + ';')  # time: 2019-10-18
        result = self.client.query('select * ' + 'from' + ' ' + war_name + ';')
        result = list(result.get_points())
        return result

    def get_svv(self, measurement: str):  # todo:time属性必须严格安装时间格式来存
        ## 返回表（一场zz）中的所有数据
        # sql = r"-- select COLUMN_NAME from information_schema.COLUMNS where table_name =" + "\"" + tableName + "\"" + " order by colorder ;"

        war_name = measurement[0] if type(measurement)==list else measurement
        result = self.client.query('select svv ' + 'from' + ' ' + war_name + ';')  # time: 2019-10-18
        result = list(result.get_points())
        return result

    def select_where(self,measurement:str, col:str, value):
        if type(measurement) == list:
            measurement = str(measurement[0])
        result = self.client.query('select * ' + 'from' + ' ' + measurement + " where \""+col+"\""+"= "+str(value)+' ;')  # time: 2019-10-18
        result = list(result.get_points())
        return result

    def insert(self, war_name, json_path):
        with open(json_path, 'rb') as f:
            params = json.load(f)
        data = Read_json(params)
        data.data_analysis()

        json_point = [
            {
                "measurement": war_name,  # 当前作战表下的
                # "tags": {
                #     "host": "server01",
                #     "region": "us-west"
                # },
                # "time": str(data.frame),
                "fields": {
                    "frame": data.frame,
                    "svv": str(data.svv),
                    "master": str(data.master),
                    "radar": str(data.radar),
                    "posx": str(data.posx),
                    "posy": str(data.posy),
                    "posz": str(data.posz),
                    "v": str(data.v),
                    "northv": str(data.northv),
                    "upv": str(data.upv),
                    "eastv": str(data.eastv),
                    "psi": str(data.psi),
                    "gv": str(data.gv),
                    "iv": str(data.iv),
                    "comm": str(data.comm)

                }
            }
        ]

        check_double_insert = self.select_where(measurement=war_name,col="frame",value=data.frame)
        if len(check_double_insert)==0:
            self.client.write_points(json_point)
        else:
            pass

    def drop(self, measurement: str):  # 删表
        self.client.drop_measurement(measurement)

    def delete(self, measurement: str, frame):  # 按帧删除: 删除大于frame的帧
        # TODO:不完善
        self.client.query("delete from " + measurement + " where frame" + " >=" + str(frame))

    def update(self,measurement,time,col):
        self.client.query(" insert" + measurement + ''+ col + 'time = '+time)

    def get_measurements(self):
        result = self.client.query("show measurements")
        measurements_list = []
        for i in range(len(list(result)[-1])):
            measurements_list.append(list(result)[-1][i]['name'])
        return measurements_list
# print(fluxdbOperator().select_num_battle('xtzz001'))
# print(fluxdbOperator().get_measurements())
# fluxdbOperator = fluxdbOperator()
# for i in fluxdbOperator.get_measurements():
#     if i.find('_')==3:
#         continue
#     fluxdbOperator.drop(i)
# drop_list = ['xtzz001','xtzz002']
# for drop_list_i in drop_list:
#     fluxdbOperator.drop(drop_list_i)
