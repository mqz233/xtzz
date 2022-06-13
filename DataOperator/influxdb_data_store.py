import json
import os
from influxdb import InfluxDBClient
from conf.readConfig import readConfig
from DataOperator.fluxdbOperator import fluxdbOperator
from DataOperator.mysqlOperator import mysqlOperator as mo
# import push
# from DataOperator.read_json import Read_json
from DataOperator.read_dynamic_json import Read_json
# from DataOperator.sensitivity_analysis import sensitivity_analysis

# current_time = datetime.datetime.utcnow().isoformat("T")
class Dynamic():
    def __init__(self, data_path):
        self.data_path = data_path

    def get_file(self, dir_path):
        list_json = []
        for root, dirs, files in os.walk(dir_path):
            # root 表示当前正在访问的文件夹路径
            # dirs 表示该文件夹下的子目录名list: 遍历所有的文件夹
            # files 表示该文件夹下的文件list
            for file in sorted(files):# 遍历文件
                # print(1)
                # for file in files:
                list_json.append(os.path.join(root, file))
        return list_json

    def store_data(self, war_name):
        assert type(war_name) == str
        list_json = self.get_file(self.data_path)
        # list_json=self.get_file(war_path)
        result_json = []
        i = 0
        j = 0
        z = 0
        o = 1
        w = 1
        number_dir = 0
        for json_path in list_json:
            if json_path.find('R') >= 0:
                continue
            with open(json_path, 'rb') as f:
                params = json.load(f)
            # break
            data = Read_json(params)
            data.new_data_analysis()  # data_analysis()
            i += 1
            if i % 60 == 0:
                j += 1
                i = 0
                if j % 60 == 0:
                    z += 1
                    j = 0

            json_point = [
                {
                    "measurement": war_name,
                    # "tags": {
                    #     "frame":str(data.frame)
                    # },
                    # "time": str(data.frame),
                    "time": "2022-5-10T" + "0" + str(z) + ":0" + str(j) + ":0" + str(i) + "Z",
                    # 每天最多存储24*60*60=86400条数据
                    "fields": {
                        # 初始数据
                        # "frame": data.frame,
                        # "svv": str(data.svv),
                        # "master": str(data.master),
                        # "radar": str(data.radar),
                        # "posx": str(data.posx),
                        # "posy": str(data.posy),
                        # "posz": str(data.posz),
                        # "v": str(data.v),
                        # "northv": str(data.northv),
                        # "upv": str(data.upv),
                        # "eastv": str(data.eastv),
                        # "psi": str(data.psi),
                        # "gv": str(data.gv),
                        # "iv": str(data.iv),
                        # "comm": str(data.comm)

                        # 新数据
                        "comm": str(data.comm),
                        "eastv": str(data.eastv),
                        "northv": str(data.northv),
                        "upv": str(data.upv),
                        "master": str(data.master),
                        "posx": str(data.posx),
                        "posy": str(data.posy),
                        "posz": str(data.posz),
                        "fTime": str(data.fTime),
                        "psi": str(data.psi),
                        "radar": str(data.radar),
                        "SuppressStatus": str(data.SuppressStatus),
                        "m_EchoStatus": str(data.m_EchoStatus),
                        "svv": str(data.svv),
                        "v": str(data.v),
                        "missileId": str(data.missileId),
                        "AttackDis": str(data.AttackDis),
                        "m_pitch": str(data.m_pitch),
                        "m_TargetNum": str(data.m_TargetNum),
                        "m_TargetList": str(data.m_TargetList),
                        "m_TargetPdList": str(data.m_TargetPdList),
                        "m_TargetAccList": str(data.m_TargetAccList),
                        "m_AngleAccList": str(data.m_AngleAccList),
                        "m_AttackNum": str(data.m_AttackNum),
                        "m_AttackList": str(data.m_AttackList),
                        "m_ControlNum": str(data.m_ControlNum),
                        "m_ControlList": str(data.m_ControlList),
                        "m_GuideID": str(data.m_GuideID)

                    }
                }
            ]

            # def show_data(client):
            #     result = client.query('select * from push_data;')
            #     print(result)

            # client = InfluxDBClient('localhost', 8086, 'root', 'root', 'LDR')  # 指定连接的数据库
            client = InfluxDBClient(str(readConfig().getInfluxdbHost()), int(readConfig().getInfluxdbPort()), 'root', 'root', str(readConfig().getInfluxdbBb()))
            client.write_points(json_point)  # 创建新表并添加数据
            # result = client.query('select * from ZZ where Frame <=10;')#各种数据库语句
            # result_json.append(result)
            # temp = pd.DataFrame(client.query('select * from students;'))
            # print(temp)#样本直接叠加存入数据库、重复样本应该覆盖而不是叠加——》主键是time+key
        number_dir += 1

    def index_data_store(self, tag, war_name): #war_nama:001/002....
        assert type(war_name) == str
        list_json = self.get_file(self.data_path)
        # list_json=self.get_file(war_path)
        result_json = []
        i = 0
        j = 0
        z = 0
        o = 1
        w = 1
        number_dir = 0
        for json_path in list_json:
            if json_path.find('R') >= 0:
                continue
            with open(json_path, 'rb') as f:
                params = json.load(f)
            # break
            data = Read_json(params)
            data.index_data_analysis()  # data_analysis()
            i += 1
            if i % 60 == 0:
                j += 1
                i = 0
                if j % 60 == 0:
                    z += 1
                    j = 0

            json_point = [
                {
                    "measurement": tag + war_name,
                    # "tags": {
                    #     "frame":str(data.frame)
                    # },
                    # "time": str(data.frame),
                    "time": "2022-5-10T" + "0" + str(z) + ":0" + str(j) + ":0" + str(i) + "Z",
                    # 每天最多存储24*60*60=86400条数据
                    "fields": {
                        "averageLev001": str(data.averageLev001),
                        "averageLev002": str(data.averageLev002),
                        "distance1": str(data.distance1),
                        "fram": str(data.fram),
                        "frdm": str(data.frdm),
                        "fttp": str(data.fttp),
                        "isSurvey": str(data.isSurvey),
                        "maxProsum": str(data.maxProsum),
                        "probedNum": str(data.probedNum),
                        "shortest_At_PathLen_Min": str(data.shortest_At_PathLen_Min),
                        "shortest_At_pre_Min": str(data.shortest_At_pre_Min),
                        "shortest_PathLen": str(data.shortest_PathLen),
                        "yjcgl": str(data.yjcgl)

                    }
                }
            ]

            # def show_data(client):
            #     result = client.query('select * from push_data;')
            #     print(result)

            # client = InfluxDBClient('localhost', 8086, 'root', 'root', 'LDR')  # 指定连接的数据库
            client = InfluxDBClient(str(readConfig().getInfluxdbHost()), int(readConfig().getInfluxdbPort()), 'root', 'root', str(readConfig().getInfluxdbBb()))
            client.write_points(json_point)  # 创建新表并添加数据
            # result = client.query('select * from ZZ where Frame <=10;')#各种数据库语句
            # result_json.append(result)
            # temp = pd.DataFrame(client.query('select * from students;'))
            # print(temp)#样本直接叠加存入数据库、重复样本应该覆盖而不是叠加——》主键是time+key
        number_dir += 1

    def index_data_store2(self, tag, war_name): #war_nama:001/002....
        assert type(war_name) == str
        list_json = self.get_file(self.data_path)
        # list_json=self.get_file(war_path)
        result_json = []
        i = 0
        j = 0
        z = 0
        o = 1
        w = 1
        number_dir = 0
        for json_path in list_json:

            with open(json_path, 'rb') as f:
                params = json.load(f)
            # break
            data = Read_json(params)
            data.index_data_analysis2()  # data_analysis()
            i += 1
            if i % 60 == 0:
                j += 1
                i = 0
                if j % 60 == 0:
                    z += 1
                    j = 0

            json_point = [
                {
                    "measurement":  'plane'+tag + war_name,
                    # "tags": {
                    #     "frame":str(data.frame)
                    # },
                    # "time": str(data.frame),
                    "time": "2022-5-10T" + "0" + str(z) + ":0" + str(j) + ":0" + str(i) + "Z",
                    # 每天最多存储24*60*60=86400条数据
                    "fields": {
                        "sences": str(data.name),
                       # "frameId": str(data.frameId),
                        "Time": str(data.time),   #########################
                        "name": str(data.name),
                        "svv": str(data.svv),
                        "frameId":str(data.frameId),
                        "isRed": str(data.isRed),
                        "type": str(data.type),
                        # "value": str(data.value),
                        # "ra_Pro_Angle": str(data.ra_Pro_Angle),
                        # "ra_Pro_Radius": str(data.ra_Pro_Radius),
                        # "ra_StartUp_Delay": str(data.ra_StartUp_Delay),
                        # "ra_Detect_Delay": str(data.ra_Detect_Delay),
                        # "ra_Process_Delay": str(data.ra_Process_Delay),
                        # "ra_FindTar_Delay": str(data.ra_FindTar_Delay),
                        # "ra_Rang_Accuracy": str(data.ra_Rang_Accuracy),
                        # "ra_Angle_Accuracy": str(data.ra_Angle_Accuracy),
                        # "MisMaxAngle": str(data.MisMaxAngle),
                        # "MisMaxRange": str(data.MisMaxRange),
                        # "MisMinDisescapeDis": str(data.MisMinDisescapeDis),
                        # "MisMaxDisescapeDis": str(data.MisMaxDisescapeDis),
                        # "MisMaxV": str(data.MisMaxV),
                        # "MisMaxOver": str(data.MisMaxOver),
                        # "MisLockTime": str(data.MisLockTime),
                        # "MisHitPro": str(data.MisHitPro),
                        # "MisMinAtkDis": str(data.MisMinAtkDis),
                        # "MisNum": str(data.MisNum),
                        # "EchoInitState": str(data.EchoInitState),
                        # "EchoFackTarNum": str(data.EchoFackTarNum),
                        # "EchoDis": str(data.EchoDis),
                        # "SupInitState": str(data.SupInitState),
                        # "SupTarNum": str(data.SupTarNum),
                        # "SupMinDis": str(data.SupMinDis),
                        # "SupMaxAngle": str(data.SupMaxAngle),
                        "posx": str(data.posx),
                        "posy": str(data.posy),
                        "posz": str(data.posz),
                        "v": str(data.v),
                        "Vn": str(data.Vn),
                        "Vu": str(data.Vu),
                        "Ve": str(data.Ve),
                        "yaw": str(data.yaw),
                        "pitch": str(data.pitch),
                        "roll": str(data.roll),
                        "radar_flag": str(data.radar_flag),
                        "rsuppress_flag": str(data.rsuppress_flag),
                        "echo_flag": str(data.echo_flag),
                        "targetNum": str(data.targetNum),
                        "radar_radius": str(data.radar_radius),
                        "atcNum": str(data.atcNum),
                        "controlNum": str(data.controlNum),
                        "comNum": str(data.comNum),
                        "suppressNum": str(data.suppressNum),
                        "echoNum": str(data.echoNum),
                        "radarList": str(data.radarList),
                        "locked": str(data.locked),
                        # "det_pro": str(data.det_pro),
                        # "range_acc": str(data.range_acc),
                        # "angle_acc": str(data.angle_acc),
                        "atkList": str(data.atkList),
                        "conList": str(data.conList),
                        "comm": str(data.comm),
                        "suppressList": str(data.suppressList),
                        "echo": str(data.echo),

                        # "stage":str(data.stage),
                        # "eval": str(data.eval)
                    }
                }
            ]

            # def show_data(client):
            #     result = client.query('select * from push_data;')
            #     print(result)

            # client = InfluxDBClient('localhost', 8086, 'root', 'root', 'LDR')  # 指定连接的数据库
            client = InfluxDBClient(str(readConfig().getInfluxdbHost()), int(readConfig().getInfluxdbPort()), 'root', 'root', str(readConfig().getInfluxdbBb()))
            client.write_points(json_point)  # 创建新表并添加数据
            # result = client.query('select * from ZZ where Frame <=10;')#各种数据库语句
            # result_json.append(result)
            # temp = pd.DataFrame(client.query('select * from students;'))
            # print(temp)#样本直接叠加存入数据库、重复样本应该覆盖而不是叠加——》主键是time+key
        number_dir += 1

    def index_store(self, tag, war_name): #war_nama:001/002....
        assert type(war_name) == str
        list_json = self.get_file(self.data_path)
        # list_json=self.get_file(war_path)
        result_json = []
        i = 0
        j = 0
        z = 0
        o = 1
        w = 1
        number_dir = 0
        for json_path in list_json:

            with open(json_path, 'rb') as f:
                params = json.load(f)
            # break
            data = Read_json(params)
            data.index_analysis()  # data_analysis()
            i += 1
            if i % 60 == 0:
                j += 1
                i = 0
                if j % 60 == 0:
                    z += 1
                    j = 0

            json_point = [
                {
                    "measurement": 'index'+ tag + war_name,
                    # "tags": {
                    #     "frame":str(data.frame)
                    # },
                    # "time": str(data.frame),
                    "time": "2022-5-10T" + "0" + str(z) + ":0" + str(j) + ":0" + str(i) + "Z",
                    # 每天最多存储24*60*60=86400条数据
                    "fields": {
                        "FackTarNum": str(data.FackTarNum),
                        "L_aver_ooda": str(data.L_aver_ooda),
                        "MisMaxAngle": str(data.MisMaxAngle),
                        "MisMaxRange": str(data.MisMaxRange),
                        "MisNum": str(data.MisNum),
                        "SupMaxAngle": str(data.SupMaxAngle),
                        "SupMinDis": str(data.SupMinDis),
                        "ThreatActionCoefAt": str(data.ThreatActionCoefAt),
                        "ThreatActionCoefCc": str(data.ThreatActionCoefCc),
                        "ThreatActionCoefCom": str(data.ThreatActionCoefCom),
                        "ThreatCoefAt": str(data.ThreatCoefAt),
                        "ThreatCoefCc": str(data.ThreatCoefCc),
                        "ThreatCoefCom": str(data.ThreatCoefCom),
                        "ability_SE_diameter": str(data.ability_SE_diameter),
                        "ability_SE_scale": str(data.ability_SE_scale),
                        "actSEdiameter": str(data.actSEdiameter),
                        "actSEscale": str(data.actSEscale),
                        "actedAdvPre": str(data.actedAdvPre),
                        "actedPre": str(data.actedPre),
                        "detectAdvPre": str(data.detectAdvPre),
                        "detectPre": str(data.detectPre),
                        "eiDistanceForTarget": str(data.eiDistanceForTarget),
                        "indexTagNum": str(data.indexTagNum),
                        "lockAdvPre": str(data.lockAdvPre),
                        "lockPre": str(data.lockPre),
                        "maxAtAbi": str(data.maxAtAbi),
                        "maxAtAbilityB": str(data.maxAtAbilityB),
                        "maxAtAbilityR": str(data.maxAtAbilityR),
                        "maxDetAbi": str(data.maxDetAbi),
                        "maxEiAbi": str(data.maxEiAbi),
                        "maxEiAbilityB": str(data.maxEiAbilityB),
                        "maxEiAbilityR": str(data.maxEiAbilityR),
                        "maxLockAngle": str(data.maxLockAngle),
                        "maxLockDistance": str(data.maxLockDistance),
                        "maxSeAbilityB": str(data.maxSeAbilityB),
                        "maxSeAbilityR": str(data.maxSeAbilityR),
                        "maxSeAngle": str(data.maxSeAngle),
                        "maxSeDistance": str(data.maxSeDistance),
                        "maxYPAbilityB": str(data.maxYPAbilityB),
                        "maxYPAbilityR": str(data.maxYPAbilityR),
                        "maxYpAbi": str(data.maxYpAbi),
                        "orgAbiAtAveB": str(data.orgAbiAtAveB),
                        "orgAbiAtAveR": str(data.orgAbiAtAveR),
                        "orgAbiAtFfAveB": str(data.orgAbiAtFfAveB),
                        "orgAbiAtFfAveR": str(data.orgAbiAtFfAveR),
                        "orgAbiEchoAveB": str(data.orgAbiEchoAveB),
                        "orgAbiEchoAveR": str(data.orgAbiEchoAveR),
                        "orgAbiEchoFfAveB": str(data.orgAbiEchoFfAveB),
                        "orgAbiEchoFfAveR": str(data.orgAbiEchoFfAveR),
                        "orgAbiEiFfAveB": str(data.orgAbiEiFfAveB),
                        "orgAbiEiFfAveR": str(data.orgAbiEiFfAveR),
                        "orgAbiLockAveB": str(data.orgAbiLockAveB),
                        "orgAbiLockAveR": str(data.orgAbiLockAveR),
                        "orgAbiSeAveB": str(data.orgAbiSeAveB),
                        "orgAbiSeAveR": str(data.orgAbiSeAveR),
                        "orgAbiSeFfAveB": str(data.orgAbiSeFfAveB),
                        "orgAbiSeFfAveR": str(data.orgAbiSeFfAveR),
                        "orgAbiSupAveB": str(data.orgAbiSupAveB),
                        "orgAbiSupAveR": str(data.orgAbiSupAveR),
                        "orgActAtAveB": str(data.orgActAtAveB),
                        "orgActAtAveR": str(data.orgActAtAveR),
                        "orgActCommAveB": str(data.orgActCommAveB),
                        "orgActCommAveR": str(data.orgActCommAveR),
                        "orgActEchoAveB": str(data.orgActEchoAveB),
                        "orgActEchoAveR": str(data.orgActEchoAveR),
                        "orgActEchoFfAveB": str(data.orgActEchoFfAveB),
                        "orgActEchoFfAveR": str(data.orgActEchoFfAveR),
                        "orgActEiFfAveB": str(data.orgActEiFfAveB),
                        "orgActEiFfAveR": str(data.orgActEiFfAveR),
                        "orgActFireFfAveB": str(data.orgActFireFfAveB),
                        "orgActFireFfAveR": str(data.orgActFireFfAveR),
                        "orgActLockAveB": str(data.orgActLockAveB),
                        "orgActLockAveR": str(data.orgActLockAveR),
                        "orgActSeAveB": str(data.orgActSeAveB),
                        "orgActSeAveR": str(data.orgActSeAveR),
                        "orgActSeFfAveB": str(data.orgActSeFfAveB),
                        "orgActSeFfAveR": str(data.orgActSeFfAveR),
                        "orgActSupAveB": str(data.orgActSupAveB),
                        "orgActSupAveR": str(data.orgActSupAveR),
                        "precisionSeAng": str(data.precisionSeAng),
                        "precisionSeDis": str(data.precisionSeDis),
                        "sencesTime": str(data.sencesTime),
                        "shortestPathLATAac": str(data.shortestPathLATAac),
                        "shortestPathLATAbi": str(data.shortestPathLATAbi),
                        "shortestPathLYPAac": str(data.shortestPathLYPAac),
                        "shortestPathLYPAbi": str(data.shortestPathLYPAbi),
                        "shortestPathLYZAac": str(data.shortestPathLYZAac),
                        "shortestPathLYZAbi": str(data.shortestPathLYZAbi),
                        "taskScaleB": str(data.taskScaleB),
                        "taskScaleR": str(data.taskScaleR),
                        "underShootPreo": str(data.underShootPreo),
                        "wasteScale": str(data.wasteScale)

                    }
                }
            ]

            # def show_data(client):
            #     result = client.query('select * from push_data;')
            #     print(result)

            # client = InfluxDBClient('localhost', 8086, 'root', 'root', 'LDR')  # 指定连接的数据库
            client = InfluxDBClient(str(readConfig().getInfluxdbHost()), int(readConfig().getInfluxdbPort()), 'root', 'root', str(readConfig().getInfluxdbBb()))
            client.write_points(json_point)  # 创建新表并添加数据
            # result = client.query('select * from ZZ where Frame <=10;')#各种数据库语句
            # result_json.append(result)
            # temp = pd.DataFrame(client.query('select * from students;'))
            # print(temp)#样本直接叠加存入数据库、重复样本应该覆盖而不是叠加——》主键是time+key
        number_dir += 1

def all_dynamic_store():
    dynamic_root_path = readConfig().getUnZipRootPath()
    path_list = os.listdir(dynamic_root_path)  # static_root_path='/root/xtzz/data/new_datas/new_init'
    path_list = sorted(path_list, key=lambda keys: [ord(i) for i in keys], reverse=False)
    for file_name in path_list:
        Dynamic(os.path.join(dynamic_root_path, file_name)).store_data('wars_'+file_name)


def dynamic_store(file_name):
    dynamic_root_path = readConfig().getDynamicRootPath()
    path = os.path.join(dynamic_root_path, file_name)
    print(path)
    Dynamic(os.path.join(dynamic_root_path, file_name)).store_data('wars_'+file_name)

# def all_index_store(tag):#taget命名：index_我们
#     # dynamic_root_path = readConfig().getUnZipRootPath()
#     dynamic_root_path = '/root/xtzz/data/new_datas/index'
#     path_list = os.listdir(dynamic_root_path)  # static_root_path='/root/xtzz/data/new_datas/new_init'
#     path_list = sorted(path_list, key=lambda keys: [ord(i) for i in keys], reverse=False)
#     for file_name in path_list:
#         Dynamic(os.path.join(dynamic_root_path, file_name)).index_data_store(tag,file_name)

def all_plane_store(tag,dynamic_root_path):#taget命名：index_我们
    # dynamic_root_path = readConfig().getUnZipRootPath()
    # dynamic_root_path = 'data/output/plane'
    filename = dynamic_root_path.split('\\')[-2]
    Dynamic(dynamic_root_path).index_data_store2(tag,filename)

def all_index_store(tag,dynamic_root_path):#taget命名：index_我们
    # dynamic_root_path = readConfig().getUnZipRootPath()
    # dynamic_root_path = 'data/output/index/1296'
    filename = dynamic_root_path.split('\\')[-2]
    Dynamic(dynamic_root_path).index_store(tag,filename)

def editpos(measurement, i, stage, eval):
    i = int(i)
    x = 0
    j = 0
    z = 0
    client = fluxdbOperator()
    r = client.select_num_battle(measurement)
    length = len(r)
    for a in range(i, i+50):
        if a > length:
            break
        x = a
        num1 = x // 36000
        num2 = x % 36000 // 3600
        num3 = x % 36000 % 3600 // 600
        num4 = x % 36000 % 3600 % 600 // 60
        num5 = x % 36000 % 3600 % 600 % 60 // 10
        num6 = x % 36000 % 3600 % 600 % 60 % 10
        result = r[a-1]

        del result['time']
        result['stage'] = str(stage)
        result['eval'] = str(eval)
        json_point = [
            {
                "measurement": measurement,
                # "tags": {
                #     "frame":str(data.frame)
                # },
                # "time": str(data.frame),
                "time": "2022-5-10T" + str(num1) + str(num2) + ":" + str(num3) + str(num4) + ":" + str(num5) + str(num6) + "Z",
                # 每天最多存储24*60*60=86400条数据
                "fields": result
            }]
        a = client.client
        a.write_points(json_point)
