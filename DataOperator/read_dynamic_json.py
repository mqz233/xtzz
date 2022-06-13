import json

from jsonpath import jsonpath


class Read_json(object):
    dict = {}

    def __init__(self, params):
        self.params = params
        # self.frame = []
        #
        # self.gv = []
        # self.iv = []
###############旧全dynamic数据##########################
        self.comm = []
        self.eastv = []
        self.northv = []
        self.upv = []
        self.master = []
        self.posx = []
        self.posy = []
        self.posz = []
        self.fTime = []  ##新属性
        self.psi = []
        self.radar = []
        self.SuppressStatus = []  ##新属性
        self.m_EchoStatus = []  ##新属性
        self.svv = []
        self.v = []
        self.missileId = []  ##新属性
        self.AttackDis = []  ##新属性
        self.m_pitch = []  ##新属性
        self.m_TargetNum = []
        self.m_TargetList = []
        self.m_TargetPdList = []
        self.m_TargetAccList = []
        self.m_AngleAccList = []
        self.m_AttackNum = []
        self.m_AttackList = []
        self.m_ControlNum = []
        self.m_ControlList = []
        self.m_GuideID = []
######################探测数据###################
        self.averageLev001 = []
        self.averageLev002 = []
        self.distance1 = []
        self.fram = []
        self.frdm = []
        self.fttp =[]
        self.isSurvey = []
        self.maxProsum = []
        self.probedNum = []
        self.shortest_At_PathLen_Min = []
        self.shortest_At_pre_Min = []
        self.shortest_PathLen = []
        self.yjcgl = []
        ###########################################plane
        self.name = []
        self.svv = []
        self.isRed = []
        self.type = []
        self.value = []
        self.ra_Pro_Angle = []
        self.ra_Pro_Radius = []
        self.ra_StartUp_Delay = []
        self.ra_Detect_Delay = []
        self.ra_Process_Delay = []
        self.ra_FindTar_Delay = []
        self.ra_Rang_Accuracy = []
        self.ra_Angle_Accuracy = []
        self.MisMaxAngle = []
        self.MisMaxRange = []
        self.MisMinDisescapeDis = []
        self.MisMaxDisescapeDis = []
        self.MisMaxV = []
        self.MisMaxOver = []
        self.MisLockTime = []
        self.MisHitPro = []
        self.MisMinAtkDis = []
        self.EchoInitState = []
        self.EchoFackTarNum = []
        self.EchoDis = []
        self.SupInitState = []
        self.SupTarNum = []
        self.SupMinDis = []
        self.posx = []
        self.posy = []
        self.posz = []
        self.V = []
        self.Vn = []
        self.Vu = []
        self.Ve = []
        self.yaw = []
        self.pitch = []
        self.roll = []
        self.radar_flag = []
        self.rsuppress_flag = []
        self.echo_flag = []
        self.targetNum = []
        self.radar_radius = []
        self.atcNum = []
        self.controlNum = []
        self.comNum = []
        self.suppressNum = []
        self.echoNum = []
        self.radarList = []
        self.locked = []
        self.det_pro = []
        self.range_acc = []
        self.angle_acc = []
        self.atkList = []
        self.conList = []
        self.comm = []
        self.suppressList = []
        self.echo = []
        ######################################index
        self.index1=[]
        self.index2=[]



    def get_json_data(self):
        with open('push_00002.json', 'rb') as f:
            params = json.load(f)
            # print(params)
            dict = params
        f.close()
        return dict

    def data_analysis(self):
        self.comm = jsonpath(self.params, '$...comm')[0]
        self.svv = jsonpath(self.params, '$..svv')[0]
        # self.master = jsonpath(self.params, '$..master')[0]
        self.radar = jsonpath(self.params, '$..radar')[0]
        self.posx = jsonpath(self.params, '$..posx')[0]
        self.posy = jsonpath(self.params, '$..posy')[0]
        # self.posz = jsonpath(self.params, '$..posz')[0]
        self.v = jsonpath(self.params, '$..v')[0]
        self.northv = jsonpath(self.params, '$..northv')[0]
        # self.upv = jsonpath(self.params, '$..upv')[0]
        self.eastv = jsonpath(self.params, '$..eastv')[0]
        # self.psi = jsonpath(self.params, '$..psi')[0]
        # self.SuppressStatus = jsonpath(self.params, '$..SuppressStatus')[0]
        # self.m_EchoStatus = jsonpath(self.params, '$..m_EchoStatus')[0]
        # self.missileId = jsonpath(self.params, '$..missileId')[0]
        # self.AttackDis = jsonpath(self.params, '$..AttackDis')[0]
        # self.m_pitch = jsonpath(self.params, '$..m_pitch')[0]
        # self.m_TargetNum = jsonpath(self.params, '$..m_TargetNum')[0]
        # self.m_TargetList = jsonpath(self.params, '$..m_TargetList')[0]
        self.m_TargetPdList = jsonpath(self.params, '$..m_TargetPdList')[0]
        # self.m_TargetAccList = jsonpath(self.params, '$..m_TargetAccList')[0]
        self.m_AttackNum = jsonpath(self.params, '$..m_AttackNum')[0]
        # self.m_AttackList = jsonpath(self.params, '$..m_AttackList')[0]
        # self.m_ControlNum = jsonpath(self.params, '$..m_ControlNum')[0]
        # self.m_ControlList = jsonpath(self.params, '$..m_ControlList')[0]
        # self.m_GuideID = jsonpath(self.params, '$..m_GuideID')[0]

        # return self.comm, self.eastv, self.northv, self.upv, self.master, self.posx, self.posy, self.posz,self.fTime, self.psi, self.radar, self.SuppressStatus, self.m_EchoStatus, self.svv, self.v, self.missileId, \
        #        self.AttackDis, self.m_pitch, self.m_TargetNum, self.m_TargetList, self.m_TargetPdList, self.m_TargetAccList, self.m_AttackNum, self.m_AttackList, self.m_ControlNum, self.m_ControlList, self.m_GuideID
        return self.comm, self.eastv, self.northv, self.posx, self.posy, self.radar, self.svv, self.v, self.m_TargetPdList, self.m_AttackNum

    def new_data_analysis(self):
        self.comm = self.params.get('plane').get('comm')
        self.svv = self.params.get('plane').get('svv')
        self.master = self.params.get('plane').get('master')
        self.radar = self.params.get('plane').get('radar')
        self.posx = self.params.get('plane').get('posx')
        self.posy = self.params.get('plane').get('posy')
        self.posz = self.params.get('plane').get('posz')
        self.v = self.params.get('plane').get('v')
        self.northv = self.params.get('plane').get('northv')
        self.upv = self.params.get('plane').get('upv')
        self.eastv = self.params.get('plane').get('eastv')
        self.psi = self.params.get('plane').get('psi')
        self.SuppressStatus = self.params.get('plane').get('SuppressStatus')
        self.m_EchoStatus = self.params.get('plane').get('m_EchoStatus')
        self.missileId = self.params.get('plane').get('missileId')
        self.AttackDis = self.params.get('plane').get('AttackDis')
        self.m_pitch = self.params.get('plane').get('m_pitch')
        self.m_TargetNum = self.params.get('plane').get('m_TargetNum')
        self.m_TargetList = self.params.get('plane').get('TargetList')
        self.m_TargetPdList = self.params.get('plane').get('m_TargetPdList')
        self.m_TargetAccList = self.params.get('plane').get('m_TargetAccList')
        self.m_AttackNum = self.params.get('plane').get('m_AttackNum')
        self.m_AttackList = self.params.get('plane').get('m_AttackList')
        self.m_ControlNum = self.params.get('plane').get('m_ControlNum')
        self.m_ControlList = self.params.get('plane').get('m_ControlList')
        self.m_GuideID = self.params.get('plane').get('m_GuideID')

        return self.comm, self.eastv, self.northv, self.upv, self.master, self.posx, self.posy, self.posz,self.fTime, self.psi, self.radar, self.SuppressStatus, self.m_EchoStatus, self.svv, self.v, self.missileId, \
               self.AttackDis, self.m_pitch, self.m_TargetNum, self.m_TargetList, self.m_TargetPdList, self.m_TargetAccList, self.m_AttackNum, self.m_AttackList, self.m_ControlNum, self.m_ControlList, self.m_GuideID
        # return self.comm, self.eastv, self.northv, self.posx, self.posy, self.radar, self.svv, self.v, self.m_TargetPdList, self.m_AttackNum

    def index_data_analysis(self):
        self.averageLev001 = self.params.get('index').get('averageLev001')
        self.averageLev002 = self.params.get('index').get('averageLev002')
        self.distance1 = self.params.get('index').get('distance1')
        self.fram = self.params.get('index').get('fram')
        self.frdm = self.params.get('index').get('frdm')
        self.fttp = self.params.get('index').get('fttp')
        self.isSurvey = self.params.get('index').get('isSurvey')
        self.maxProsum = self.params.get('index').get('maxProsum')
        self.probedNum = self.params.get('index').get('probedNum')
        self.shortest_At_PathLen_Min = self.params.get('index').get('shortest_At_PathLen_Min')
        self.shortest_At_pre_Min = self.params.get('index').get('shortest_At_pre_Min')
        self.shortest_PathLen = self.params.get('index').get('shortest_PathLen')
        self.yjcgl = self.params.get('index').get('yjcgl')

        return self.averageLev001, self.averageLev002, self.distance1, self.fram, self.frdm, self.fttp, self.isSurvey, self.maxProsum, \
               self.probedNum, self.shortest_At_PathLen_Min, self.shortest_At_pre_Min, self.shortest_PathLen, self.yjcgl


    def index_data_analysis2(self):
        self.sences = self.params.get('plane').get('sences')
        self.frameId = self.params.get('plane').get('frameId')
        self.time = self.params.get('plane').get('time')
        self.name = self.params.get('plane').get('name')
        self.svv = self.params.get('plane').get('svv')
        self.isRed = self.params.get('plane').get('isRed')
        self.type = self.params.get('plane').get('type')
        self.value = self.params.get('plane').get('value')
        self.ra_Pro_Angle = self.params.get('plane').get('ra_Pro_Angle')
        self.ra_Pro_Radius = self.params.get('plane').get('ra_Pro_Radius')
        self.ra_StartUp_Delay = self.params.get('plane').get('ra_StartUp_Delay')
        self.ra_Detect_Delay = self.params.get('plane').get('ra_Detect_Delay')
        self.ra_Process_Delay = self.params.get('plane').get('ra_Process_Delay')
        self.ra_FindTar_Delay = self.params.get('plane').get('ra_FindTar_Delay')
        self.ra_Rang_Accuracy = self.params.get('plane').get('ra_Rang_Accuracy')
        self.ra_Angle_Accuracy = self.params.get('plane').get('ra_Angle_Accuracy')
        self.MisMaxAngle = self.params.get('plane').get('MisMaxAngle')
        self.MisMaxRange = self.params.get('plane').get('MisMaxRange')
        self.MisMinDisescapeDis = self.params.get('plane').get('MisMinDisescapeDis')
        self.MisMaxDisescapeDis = self.params.get('plane').get('MisMaxDisescapeDis')
        self.MisMaxV = self.params.get('plane').get('MisMaxV')
        self.MisMaxOver = self.params.get('plane').get('MisMaxOver')
        self.MisLockTime = self.params.get('plane').get('MisLockTime')
        self.MisHitPro = self.params.get('plane').get('MisHitPro')
        self.MisMinAtkDis = self.params.get('plane').get('MisMinAtkDis')
        self.MisNum = self.params.get('plane').get('MisNum')
        self.EchoInitState = self.params.get('plane').get('EchoInitState')
        self.EchoFackTarNum = self.params.get('plane').get('EchoFackTarNum')
        self.EchoDis = self.params.get('plane').get('EchoDis')
        self.SupInitState = self.params.get('plane').get('SupInitState')
        self.SupTarNum = self.params.get('plane').get('SupTarNum')
        self.SupMinDis = self.params.get('plane').get('SupMinDis')
        self.SupMaxAngle = self.params.get('plane').get('SupMaxAngle')
        self.posx = self.params.get('plane').get('posx')
        self.posy = self.params.get('plane').get('posy')
        self.posz = self.params.get('plane').get('posz')
        self.v = self.params.get('plane').get('v')
        self.Vn = self.params.get('plane').get('Vn')
        self.Vu = self.params.get('plane').get('Vu')
        self.Ve = self.params.get('plane').get('Ve')
        self.yaw = self.params.get('plane').get('yaw')
        self.pitch = self.params.get('plane').get('pitch')
        self.roll = self.params.get('plane').get('roll')
        self.radar_flag = self.params.get('plane').get('radar_flag')
        self.rsuppress_flag = self.params.get('plane').get('rsuppress_flag')
        self.echo_flag = self.params.get('plane').get('echo_flag')
        self.targetNum = self.params.get('plane').get('targetNum')
        self.radar_radius = self.params.get('plane').get('radar_radius')
        self.atcNum = self.params.get('plane').get('atcNum')
        self.controlNum = self.params.get('plane').get('controlNum')
        self.comNum = self.params.get('plane').get('comNum')
        self.suppressNum = self.params.get('plane').get('suppressNum')
        self.echoNum = self.params.get('plane').get('echoNum')
        self.radarList = self.params.get('plane').get('radarList')
        self.locked = self.params.get('plane').get('locked')
        self.det_pro = self.params.get('plane').get('det_pro')
        self.range_acc = self.params.get('plane').get('range_acc')
        self.angle_acc = self.params.get('plane').get('angle_acc')
        self.atkList = self.params.get('plane').get('atkList')
        self.conList = self.params.get('plane').get('conList')
        self.comm = self.params.get('plane').get('comm')
        self.suppressList = self.params.get('plane').get('suppressList')
        self.echo = self.params.get('plane').get('echo')

        # self.stage = ''
        # self.eval = ''


    def index_analysis(self):
        self.FackTarNum = self.params.get('index').get('FackTarNum')
        self.L_aver_ooda = self.params.get('index').get('L_aver_ooda')
        self.MisMaxAngle = self.params.get('index').get('MisMaxAngle')
        self.MisMaxRange = self.params.get('index').get('MisMaxRange')
        self.MisNum = self.params.get('index').get('MisNum')
        self.SupMaxAngle = self.params.get('index').get('SupMaxAngle')
        self.SupMinDis = self.params.get('index').get('SupMinDis')
        self.ThreatActionCoefAt = self.params.get('index').get('ThreatActionCoefAt')
        self.ThreatActionCoefCc = self.params.get('index').get('ThreatActionCoefCc')
        self.ThreatActionCoefCom = self.params.get('index').get('ThreatActionCoefCom')
        self.ThreatCoefAt = self.params.get('index').get('ThreatCoefAt')
        self.ThreatCoefCc = self.params.get('index').get('ThreatCoefCc')
        self.ThreatCoefCom = self.params.get('index').get('ThreatCoefCom')
        self.ability_SE_diameter = self.params.get('index').get('ability_SE_diameter')
        self.ability_SE_scale = self.params.get('index').get('ability_SE_scale')
        self.actSEdiameter = self.params.get('index').get('actSEdiameter')
        self.actSEscale = self.params.get('index').get('actSEscale')
        self.actedAdvPre = self.params.get('index').get('actedAdvPre')
        self.actedPre = self.params.get('index').get('actedPre')
        self.detectAdvPre = self.params.get('index').get('detectAdvPre')
        self.detectPre = self.params.get('index').get('detectPre')
        self.eiDistanceForTarget = self.params.get('index').get('eiDistanceForTarget')
        self.indexTagNum = self.params.get('index').get('indexTagNum')
        self.lockAdvPre = self.params.get('index').get('lockAdvPre')
        self.lockPre = self.params.get('index').get('lockPre')
        self.maxAtAbi = self.params.get('index').get('maxAtAbi')
        self.maxAtAbilityB = self.params.get('index').get('maxAtAbilityB')
        self.maxAtAbilityR = self.params.get('index').get('maxAtAbilityR')
        self.maxDetAbi = self.params.get('index').get('maxDetAbi')
        self.maxEiAbi = self.params.get('index').get('maxEiAbi')
        self.maxEiAbilityB = self.params.get('index').get('maxEiAbilityB')
        self.maxEiAbilityR = self.params.get('index').get('maxEiAbilityR')
        self.maxLockAngle = self.params.get('index').get('maxLockAngle')
        self.maxLockDistance = self.params.get('index').get('maxLockDistance')
        self.maxSeAbilityB = self.params.get('index').get('maxSeAbilityB')
        self.maxSeAbilityR = self.params.get('index').get('maxSeAbilityR')
        self.maxSeAngle = self.params.get('index').get('maxSeAngle')
        self.maxSeDistance = self.params.get('index').get('maxSeDistance')
        self.maxYPAbilityB = self.params.get('index').get('maxYPAbilityB')
        self.maxYPAbilityR = self.params.get('index').get('maxYPAbilityR')
        self.maxYpAbi = self.params.get('index').get('maxYpAbi')
        self.orgAbiAtAveB = self.params.get('index').get('orgAbiAtAveB')
        self.orgAbiAtAveR = self.params.get('index').get('orgAbiAtAveR')
        self.orgAbiAtFfAveB = self.params.get('index').get('orgAbiAtFfAveB')
        self.orgAbiAtFfAveR = self.params.get('index').get('orgAbiAtFfAveR')
        self.orgAbiEchoAveB = self.params.get('index').get('orgAbiEchoAveB')
        self.orgAbiEchoAveR = self.params.get('index').get('orgAbiEchoAveR')
        self.orgAbiEchoFfAveB = self.params.get('index').get('orgAbiEchoFfAveB')
        self.orgAbiEchoFfAveR = self.params.get('index').get('orgAbiEchoFfAveR')
        self.orgAbiEiFfAveB = self.params.get('index').get('orgAbiEiFfAveB')
        self.orgAbiEiFfAveR = self.params.get('index').get('orgAbiEiFfAveR')
        self.orgAbiLockAveB = self.params.get('index').get('orgAbiLockAveB')
        self.orgAbiLockAveR = self.params.get('index').get('orgAbiLockAveR')
        self.orgAbiSeAveB = self.params.get('index').get('orgAbiSeAveB')
        self.orgAbiSeAveR = self.params.get('index').get('orgAbiSeAveR')
        self.orgAbiSeFfAveB = self.params.get('index').get('orgAbiSeFfAveB')
        self.orgAbiSeFfAveR = self.params.get('index').get('orgAbiSeFfAveR')
        self.orgAbiSupAveB = self.params.get('index').get('orgAbiSupAveB')
        self.orgAbiSupAveR = self.params.get('index').get('orgAbiSupAveR')
        self.orgActAtAveB = self.params.get('index').get('orgActAtAveB')
        self.orgActAtAveR = self.params.get('index').get('orgActAtAveR')
        self.orgActCommAveB = self.params.get('index').get('orgActCommAveB')
        self.orgActCommAveR = self.params.get('index').get('orgActCommAveR')
        self.orgActEchoAveB = self.params.get('index').get('orgActEchoAveB')
        self.orgActEchoAveR = self.params.get('index').get('orgActEchoAveR')
        self.orgActEchoFfAveB = self.params.get('index').get('orgActEchoFfAveB')
        self.orgActEchoFfAveR = self.params.get('index').get('orgActEchoFfAveR')
        self.orgActEiFfAveB = self.params.get('index').get('orgActEiFfAveB')
        self.orgActEiFfAveR = self.params.get('index').get('orgActEiFfAveR')
        self.orgActFireFfAveB = self.params.get('index').get('orgActFireFfAveB')
        self.orgActFireFfAveR = self.params.get('index').get('orgActFireFfAveR')
        self.orgActLockAveB = self.params.get('index').get('orgActLockAveB')
        self.orgActLockAveR = self.params.get('index').get('orgActLockAveR')
        self.orgActSeAveB = self.params.get('index').get('orgActSeAveB')
        self.orgActSeAveR = self.params.get('index').get('orgActSeAveR')
        self.orgActSeFfAveB = self.params.get('index').get('orgActSeFfAveB')
        self.orgActSeFfAveR = self.params.get('index').get('orgActSeFfAveR')
        self.orgActSupAveB = self.params.get('index').get('orgActSupAveB')
        self.orgActSupAveR = self.params.get('index').get('orgActSupAveR')
        self.precisionSeAng = self.params.get('index').get('precisionSeAng')
        self.precisionSeDis = self.params.get('index').get('precisionSeDis')
        self.sencesTime = self.params.get('index').get('sencesTime')
        self.shortestPathLATAac = self.params.get('index').get('shortestPathLATAac')
        self.shortestPathLATAbi = self.params.get('index').get('shortestPathLATAbi')
        self.shortestPathLYPAac = self.params.get('index').get('shortestPathLYPAac')
        self.shortestPathLYPAbi = self.params.get('index').get('shortestPathLYPAbi')
        self.shortestPathLYZAac = self.params.get('index').get('shortestPathLYZAac')
        self.shortestPathLYZAbi = self.params.get('index').get('shortestPathLYZAbi')
        self.taskScaleB = self.params.get('index').get('taskScaleB')
        self.taskScaleR = self.params.get('index').get('taskScaleR')
        self.underShootPreo = self.params.get('index').get('underShootPreo')
        self.wasteScale = self.params.get('index').get('wasteScale')
