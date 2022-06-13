import datetime
import json
import os

from influxdb import InfluxDBClient

# import push
from read_json import Read_json

current_time = datetime.datetime.utcnow().isoformat("T")


def get_file(dir_path):
    list_json = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            list_json.append(os.path.join(root, file))
    return list_json


list_json = get_file(r".\push")
result_json = []
i = 0
j = 0
z = 0
for json_path in list_json:
    with open(json_path, 'rb') as f:
        params = json.load(f)
    data = Read_json(params)
    data.data_analysis()
    i += 1
    if i % 60 == 0:
        j += 1
        i = 0
        if j % 60 == 0:
            z += 1
            j = 0
    json_point = [
        {
            # todo:改为"json_output_5"+str(i)+"255555", + 一个文件夹所有文件
            "measurement": "json_output_5355555",
            # "tags": {
            #     "host": "server01",
            #     "region": "us-west"
            # },
            # "time": str(data.frame), #TODO:用i,j,z的增加表示时间序列 √
            "time": "2021-10-13T" + "0" + str(z) + ":0" + str(j) + ":0" + str(i) + "Z",  # 每天最多存储24*60*60=86400条数据
            # "time": "2021-10-12T",
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

    # def show_data(client):
    #     result = client.query('select * from push_data;')
    #     print(result)

    # client = InfluxDBClient('localhost', 8086, 'root', 'root', 'LDR')  # 指定连接的数据库
    # client.write_points(json_point)  # 创建新表并添加数据


    # result = client.query('select * from ZZ where Frame <=10;')#各种数据库语句
    # result_json.append(result)
    # temp = pd.DataFrame(client.query('select * from students;'))
    # print(temp)#样本直接叠加存入数据库、重复样本应该覆盖而不是叠加——》主键是time+key

# print(result_json)
