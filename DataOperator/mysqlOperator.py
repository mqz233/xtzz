import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "djangoProject.settings")
django.setup()
from app0 import models
from app0.models import Static_war_data, Init_1207 ,Last_1207
from app0.models import Static_platform_data
from app0.models import InitialFrame, AttackLast, AttackInit,RepressInit,RepressLast
from app0.models import Frame, Index
from conf.readConfig import readConfig as rc
from DataOperator.jsonOperator import jsonOperator as jo



class mysqlOperator():
    def __init__(self):
        self.dataRootDirPath = rc().getDataRootDirPath()

    def insertStaticDataByWarPath(self, war_name, warPath=''):
        # war_name = os.path.split(warPath)[1]
        # dir = os.path.join(warPath,'R') ## Json 文件列表
        jsonPathList = os.listdir(warPath)
        jsonPath = ''
        for i in jsonPathList:
            if i.find('R') >= 0:
                jsonPath = os.path.join(warPath, i)
                break
        # jsonPath  = warPath
        warStaticData = jo().convertJsonToDict(jsonPath)
        # warStaticData=warStaticData[0]
        if not self.checkStaticData(war_name):
            Static_war_data(
                # id =
                war_name=war_name,
                version=warStaticData['version'],
                posmode=warStaticData['posmode'],
                platforms=len(warStaticData['platforms']),
                comm=warStaticData['comm']
            ).save()

        platformStaticDatas = warStaticData['platforms']
        for platformStaticData in platformStaticDatas:
            if not self.checkPlatformData(war_name=war_name, uav_id=platformStaticData['id']):
                Static_platform_data(
                    war_name=war_name,
                    uav_id=platformStaticData['id'],
                    svv=platformStaticData['svv'],
                    type=platformStaticData['type'],
                    master=platformStaticData['master'],
                    radar=platformStaticData['radar'],
                    fram=platformStaticData['fram'],
                    frdm=platformStaticData['frdm'],
                    fmdm=platformStaticData['fmdm'],
                    fmam=platformStaticData['fmam'],
                    fmdkmax=platformStaticData['fmdkmax'],
                    fmdkmin=platformStaticData['fmdkmin'],
                    lng=platformStaticData['lng'],
                    lat=platformStaticData['lat'],
                    height=platformStaticData['height'],
                    posx=platformStaticData['posx'],
                    posy=platformStaticData['posy'],
                    posz=platformStaticData['posz'],
                    v=platformStaticData['v'],
                    northv=platformStaticData['northv'],
                    upv=platformStaticData['upv'],
                    eastv=platformStaticData['eastv'],
                    iv=platformStaticData['iv'],
                    gv=platformStaticData['gv']
                ).save()

    def insertStaticDataFromJson(self, war_name, json_location):
        json_data = jo().convertJsonToDict(json_location)

        ## 修改全局 zz 结构，存入数据库
        platform_dicts = json_data['platforms']
        json_data['platforms'] = len(platform_dicts)
        json_data['war_name'] = war_name

        # 保存全局 zz 数据
        if not self.checkStaticData(war_name):
            Static_war_data(
                # id =
                war_name=json_data['war_name'],
                version=json_data['version'],
                posmode=json_data['posmode'],
                platforms=json_data['platforms'],
                comm=json_data['comm']
            ).save()

        ## 修改平台数据，存入platform数据库
        for platform_i in platform_dicts:
            ## 修改数据结构
            platform_i['war_name'] = war_name
            platform_i['uav_id'] = platform_i['id']
            del platform_i['id']

            ## 存入数据库
            if not self.checkPlatformData(war_name=war_name, uav_id=platform_i['uav_id']):
                Static_platform_data(
                    war_name=platform_i['war_name'],
                    uav_id=platform_i['uav_id'],
                    svv=platform_i['svv'],
                    type=platform_i['type'],
                    master=platform_i['master'],
                    radar=platform_i['radar'],
                    fram=platform_i['fram'],
                    frdm=platform_i['frdm'],
                    fmdm=platform_i['fmdm'],
                    fmam=platform_i['fmam'],
                    fmdkmax=platform_i['fmdkmax'],
                    fmdkmin=platform_i['fmdkmin'],
                    lng=platform_i['lng'],
                    lat=platform_i['lat'],
                    height=platform_i['height'],
                    posx=platform_i['posx'],
                    posy=platform_i['posy'],
                    posz=platform_i['posz'],
                    v=platform_i['v'],
                    northv=platform_i['northv'],
                    upv=platform_i['upv'],
                    eastv=platform_i['eastv'],
                    iv=platform_i['iv'],
                    gv=platform_i['gv']
                ).save()

    def insert_single_data_from_dict(self, table_name, values):
        ## values: 要输入的一个数组，

        if table_name == 'InitialFrame':
            InitialFrame(
                posmode=values['posmode'],
                SencesNum=values['SencesNum'],
                name=values['name'],
                type=values['type'],
                fram=values['fram'],
                frdm=values['frdm'],
                sdelay=values['sdelay'],
                pdelay=values['pdelay'],
                diff=values['diff'],
                fmam=values['fmam'],
                fmdm=values['fmdm'],
                fmdkmax=values['fmdkmax'],
                fmdkmin=values['fmdkmin'],
                isRed=values['isRed'],
                echo=values['echo'],
                fsjtn=values['fsjtn'],
                fetn=values['fetn'],
                fmv=values['fmv'],
                fmload=values['fmload'],
                ftdt=values['ftdt'],
                fttp=values['fttp'],
                fsuppress=values['fsuppress'],
                fsjdmin=values['fsjdmin'],
                fdjdm=values['fdjdm'],
                fmtlt=values['fmtlt'],
                fmcp=values['fmcp'],
                fmhp=values['fmhp'],
                value=values['value']
            ).save()

    def insert_single_tag_data_from_dict(self, table_name, values):
        ## values: 要输入的一个数组，

        if table_name == 'Init_1207':
            Init_1207(
                tagid=values['tagid'],
                posmode=values['posmode'],
                SencesNum=values['SencesNum'],
                name=values['name'],
                type=values['type'],
                fram=values['fram'],
                frdm=values['frdm'],
                fmam=values['fmam'],
                fmdm=values['fmdm'],
                isRed=values['isRed'],
                fsjtn=values['fsjtn'],
                fetn=values['fetn'],
                fttp=values['fttp'],
                fsjdmin=values['fsjdmin'],
                fdjdm=values['fdjdm'],
                fmcp=values['fmcp'],
                fmhp=values['fmhp'],
                fmhdmin=values['fmhdmin']
            ).save()

        if table_name == 'Last_1207':
            Last_1207(
                tagid=values['tagid'],
                SENODE2ofEdge=values['SENODE2ofEdge'],
                SENODEofEdge=values['SENODEofEdge'],
                aveDistance1=values['aveDistance1'],
                averageLev001=values['averageLev001'],
                averageLev002=values['averageLev002'],
                averageLev003=values['averageLev003'],
                averageLev004=values['averageLev004'],
                averageLev2=values['averageLev2'],
                betweenNess3=values['betweenNess3'],
                betweenNess4=values['betweenNess4'],
                distance1=values['distance1'],
                fram=values['fram'],
                frdm=values['frdm'],
                fttp=values['fttp'],
                isSurvey=values['isSurvey'],
                maxProsum=values['maxProsum'],
                mbsbzql=values['mbsbzql'],
                mbtcxxffblhfsj=values['mbtcxxffblhfsj'],
                mbtcxxffzjhfsj=values['mbtcxxffzjhfsj'],
                probedNum=values['probedNum'],
                shortest_SeXW_PathLen_Min=values['shortest_SeXW_PathLen_Min'],
                shortest_SeXW_pre_Min=values['shortest_SeXW_pre_Min'],
                shortest_SeYS_PathLen_Min=values['shortest_SeYS_PathLen_Min'],
                shortest_SeYS_pre_Min=values['shortest_SeYS_pre_Min'],
                tchfsj=values['tchfsj'],
                xdfxjl=values['xdfxjl'],
                xdfxsj=values['xdfxsj'],
                yjcgl=values['yjcgl']
            ).save()

        if table_name == 'AttackInit':
            AttackInit(
                tagid=values['tagid'],
                posmode=values['posmode'],
                SencesNum=values['SencesNum'],
                name=values['name'],
                type=values['type'],
                fram=values['fram'],
                frdm=values['frdm'],
                fmam=values['fmam'],
                fmdm=values['fmdm'],
                isRed=values['isRed'],
                fsjtn=values['fsjtn'],
                fetn=values['fetn'],
                fmv=values['fmv'],
                fmload=values['fmload'],
                ftdt=values['ftdt'],
                fttp=values['fttp'],
                fsjdmin=values['fsjdmin'],
                fdjdm=values['fdjdm'],
                fmtlt=values['fmtlt'],
                fmcp=values['fmcp'],
                fmhp=values['fmhp'],
                missile_num=values['missile_num'],
                fmhdmin=values['fmhdmin']
            ).save()

        if table_name == 'AttackLast':
            AttackLast(
                tagid=values['tagid'],
                AttackDis=values['AttackDis'],
                SENODE2ofEdge=values['SENODE2ofEdge'],
                SENODEofEdge=values['SENODEofEdge'],
                aveDistance1=values['aveDistance1'],
                averageLev001=values['averageLev001'],
                averageLev002=values['averageLev002'],
                averageLev003=values['averageLev003'],
                averageLev004=values['averageLev004'],
                averageLev2=values['averageLev2'],
                betweenNess3=values['betweenNess3'],
                betweenNess4=values['betweenNess4'],
                distance1=values['distance1'],
                fmam=values['fmam'],
                fmdm=values['fmdm'],
                fram=values['fram'],
                frdm=values['frdm'],
                fttp=values['fttp'],
                isSurvey=values['isSurvey'],
                maxAtcsum=values['maxAtcsum'],
                maxProsum=values['maxProsum'],
                mbjhxxffzdlj=values['mbjhxxffzdlj'],
                mbjhxxffzdljhfsj=values['mbjhxxffzdljhfsj'],
                mbsbzql=values['mbsbzql'],
                mbtcxxffblhfsj=values['mbtcxxffblhfsj'],
                mbtcxxffzjhfsj=values['mbtcxxffzjhfsj'],
                missile_num=values['missile_num'],
                probedNum=values['probedNum'],
                shortestAtXWMzxVal=values['shortestAtXWMzxVal'],
                shortestAtYSMzxVal=values['shortestAtYSMzxVal'],
                shortest_SeXW_pre_Min=values['shortest_SeXW_pre_Min'],
                shortest_SeYS_pre_Min=values['shortest_SeYS_pre_Min'],
                tchfsj=values['tchfsj'],
                xdfsjl=values['xdfsjl'],
                xdfssj=values['xdfssj'],
                xdfxjl=values['xdfxjl'],
                xdfxsj=values['xdfxsj'],
                yjcgl=values['yjcgl']
            ).save()

        if table_name == 'RepressInit':
            RepressInit(
                tagid=values['tagid'],
                posmode=values['posmode'],
                SencesNum=values['SencesNum'],
                name=values['name'],
                type=values['type'],
                fram=values['fram'],
                frdm=values['frdm'],
                fmam=values['fmam'],
                fmdm=values['fmdm'],
                isRed=values['isRed'],
                fsjtn=values['fsjtn'],
                fetn=values['fetn'],
                fmv=values['fmv'],
                fmload =values['fmload'],
                ftdt=values['ftdt'],
                fttp=values['fttp'],
                fsjdmin=values['fsjdmin'],
                fdjdm=values['fdjdm'],
                fmtlt=values['fmtlt'],
                fmcp=values['fmcp'],
                fmhp=values['fmhp'],
                missile_num=values['missile_num'],
                fmhdmin=values['fmhdmin']
            ).save()


        if table_name == 'RepressLast':
            RepressLast(
                tagid=values['tagid'],
                AttackDis=values['AttackDis'],
                SENODE2ofEdge=values['SENODE2ofEdge'],
                SENODEofEdge=values['SENODEofEdge'],
                aveDistance1=values['aveDistance1'],
                averageLev001=values['averageLev001'],
                averageLev002=values['averageLev002'],
                averageLev003=values['averageLev003'],
                averageLev004=values['averageLev004'],
                averageLev2=values['averageLev2'],
                betweenNess3=values['betweenNess3'],
                betweenNess4=values['betweenNess4'],
                distance1=values['distance1'],
                fetn=values['fetn'],
                fmam=values['fmam'],
                fmdm=values['fmdm'],
                fram=values['fram'],
                frdm=values['frdm'],
                fttp=values['fttp'],
                isSurvey=values['isSurvey'],
                m_echoNum=values['m_echoNum'],
                maxEicsum=values['maxEicsum'],
                maxProsum=values['maxProsum'],
                maxYPcsum=values['maxYPcsum'],
                mbjhxxffzdlj=values['mbjhxxffzdlj'],
                mbjhxxffzdljhfsj=values['mbjhxxffzdljhfsj'],
                mbsbzql=values['mbsbzql'],
                mbtcxxffblhfsj=values['mbtcxxffblhfsj'],
                mbtcxxffzjhfsj=values['mbtcxxffzjhfsj'],
                missile_num=values['missile_num'],
                probedNum=values['probedNum'],
                shortestAtXWMzxVal=values['shortestAtXWMzxVal'],
                shortestEIXWMzxVal=values['shortestEIXWMzxVal'],
                shortest_SeXW_pre_Min=values['shortest_SeXW_pre_Min'],
                shortest_SeYS_pre_Min=values['shortest_SeYS_pre_Min'],
                tchfsj=values['tchfsj'],
                xdfsjl=values['xdfsjl'],
                xdfssj=values['xdfssj'],
                xdfxjl=values['xdfxjl'],
                xdfxsj=values['xdfxsj'],
                yjcgl=values['yjcgl']
            ).save()

        if table_name == 'Frame':
            Frame(
                tagid=values['tagid'],
                SencesNum=values['SencesNum'],
                name=values['name'],
                isRed=values['isRed'],
                type=values['type'],
                value=values['value'],
                ra_Pro_Angle=values['ra_Pro_Angle'],
                ra_Probe_Radius=values['ra_Probe_Radius'],
                ra_StartUp_Delay=values['ra_StartUp_Delay'],
                ra_Process_Delay=values['ra_Process_Delay'],
                ra_Rang_Accuracy=values['ra_Rang_Accuracy'],
                ra_Angle_Accuracy=values['ra_Angle_Accuracy'],
                MisMaxAngle=values['MisMaxAngle'],
                MisMinDisescapeDis=values['MisMinDisescapeDis'],
                MisMaxDisescapeDis=values['MisMaxDisescapeDis'],
                MisMaxV=values['MisMaxV'],
                MisMaxRange=values['MisMaxRange'],
                MisHitPro=values['MisHitPro'],
                ra_Detect_Delay=values['ra_Detect_Delay'],
                ra_FindTar_Delay=values['ra_FindTar_Delay'],
                MisMaxOver=values['MisMaxOver'],
                MisLockTime=values['MisLockTime'],
                MisMinAtkDis=values['MisMinAtkDis'],
                EchoInitState=values['EchoInitState'],
                EchoFackTarNum=values['EchoFackTarNum'],
                EchoDis=values['EchoDis'],
                SupInitState=values['SupInitState'],
                SupTarNum=values['SupTarNum'],
                SupMinDis=values['SupMinDis'],
                MisNum=values['MisNum'],
                SupMaxAngle=values['SupMaxAngle'],
                planeNum = values['planeNum'],
                warname = values['warname']
            ).save()

        if table_name == 'Index':
            Index(
                tagid=values['tagid'],
                # findDistance=values['findDistance'],
                # findTime=values['findTime'],
                frameNum=values['frameNum'],
                tarDetSta=values['tarDetSta'],
                timeBeforAt=values['timeBeforAt'],
                # timeBeforEi=values['timeBeforEi'],
                # resumeTimeDt=values['resumeTimeDt'],
                redMisHitRate=values['redMisHitRate'],
                rateOfSucEarlyWaring=values['rateOfSucEarlyWaring'],
                # distanceBeforEi=values['distanceBeforEi'],
                distanceBeforAt=values['distanceBeforAt'],
                blueMisHitRate=values['blueMisHitRate'],
                aveDistanceFirstFindRed=values['aveDistanceFirstFindRed'],
                aveDistanceFirstFindBlue=values['aveDistanceFirstFindBlue'],
            disWhenAt = values['disWhenAt'],
            distanceBeforLock = values['distanceBeforLock'],
            distanceBeforSe = values['distanceBeforSe'],
            resumeTimeAt = values['resumeTimeAt'],
            resumeTimeDtAbi = values['resumeTimeDtAbi'],
            resumeTimeDtAct = values['resumeTimeDtAct'],
            resumeTimeDtPropor = values['resumeTimeDtPropor'],
            resumeTimeEi = values['resumeTimeEi'],
            resumeTimeFake = values['resumeTimeFake'],
            resumeTimeLock = values['resumeTimeLock'],
            resumeTimeShortestEi = values['resumeTimeShortestEi'],
            resumeTimeShortestFake = values['resumeTimeShortestFake'],
            resumeTimeShortestLock = values['resumeTimeShortestLock'],
            tarLockSta = values['tarLockSta'],
            timeBeforLock = values['timeBeforLock'],
            timeBeforSe = values['timeBeforSe'],
            shape = values['shape'],
            shapeCoeff = values['shapeCoeff'],
                warname=values['warname']
            ).save()


    def checkStaticData(self, war_name):
        ## 检查当前静态zz 数据是否已经导入数据库,已导入返回True 不允许导入，否则为False
        war_name_list = self.queryWarData()
        if war_name in war_name_list:
            return True
        else:
            return False

    def checkPlatformData(self, war_name, uav_id):
        ## 检查当前平台是否已经导入数据库， 如果已经导入，则为True
        results = Static_platform_data.objects.filter(war_name=war_name, uav_id=uav_id).values()
        if len(results) > 0:
            return True
        else:
            return False

    def queryStaticData(self, requiredData):
        """
        从MySQL数据库中读出需要的数据
        :param requiredData: [war_name,col1,col2...]
        :return: queryset
        """
        ## filter == where, values == select
        len_req_data = len(requiredData)
        # current_war_name = requiredData[0]['war_name']
        current_war_name = requiredData[0]
        war_data = Static_war_data.objects.filter(war_name=current_war_name).values()  ## 只有一条数据
        platform_data = Static_platform_data.objects.filter(war_name=current_war_name).values(*requiredData[1:])
        # initialFrame = InitialFrame.objects.filter(war_name=current_war_name).values()
        return {'war_data':war_data, 'platform_data':platform_data}

    def queryNewStaticData(self, requiredData:int):
        """
        从MySQL数据库中读出需要的数据
        :param requiredData: [war_name,col1,col2...]
        :return: queryset
        """
        ## filter == where, values == select
        # current_war_name = requiredData[0]['war_name']
        current_war_name = requiredData
        initialFrame = InitialFrame.objects.filter(SencesNum=current_war_name).values()
        return {'initialFrame':initialFrame}

    def queryNewTagStaticData(self, requiredData:str):#精确查询
        """
        从MySQL数据库中读出需要的数据
        :param requiredData: [war_name,col1,col2...]
        :return: queryset
        """
        ## filter == where, values == select
        # current_war_name = requiredData[0]['war_name']
        tagid = requiredData
        Frame = list(models.Frame.objects.filter(tagid=tagid).values())
        Index = list(models.Index.objects.filter(tagid=tagid).values())
        return {'Frame':Frame, 'Index':Index}

    def FuzzyQueryNewTagStaticData(self, requiredData:str): #模糊查询：把所有tagid中tag是requiredData的全部数据查询出来
        tag = requiredData
        init_1207 = Init_1207.objects.filter(tagid__startswith=tag).values()
        last_1207 = Last_1207.objects.filter(tagid__startswith=tag).values()
        return {'Init_1207': init_1207, 'Last_1207': last_1207}

    def QueryAttackTagStaticData(self, requiredData:str):
        tagid = requiredData
        attackInit = AttackInit.objects.filter(tagid=tagid).values()
        attackLast = AttackLast.objects.filter(tagid=tagid).values()
        return {'AttackInit': attackInit, 'AttackLast': attackLast}

    def getAllStaticFeatures(self):
        ## 返回需要展示的静态指标列表
        return ['uav_id', 'svv', 'type', 'master', 'radar', 'type', 'fram', 'frdm', 'sdelay', 'pdelay', 'fmam', 'fmdm',
                'fmdkmax', 'fmdkmin', 'lng', 'lat', 'height', 'posx', 'posy', 'posz', 'v', 'northv', 'upv', 'eastv',
                'psi', 'gv', 'iv']

    def getValuableStaticFeatures(self):
        ## 返回 13 个有价值的静态值，包括svv
        return ['svv', 'type', 'fram', 'frdm', 'sdelay', 'pdelay', 'fmam', 'fmdm', 'fmdkmax', 'fmdkmin', 'lng', 'lat',
                'height']

    def queryWarData(self):
        ## 查询所有作战文件夹名称
        war_data = Static_war_data.objects.values('war_name')
        return war_data

    def queryTag(self):
        tag = Frame.objects.values('tagid')
        tag = list(tag)
        tag_list = []
        for tag_i in tag:
            tag_list.append(tag_i['tagid'])
        return list(set(tag_list))#元素去重

    def querywarname(self,tag):
        a = list(Frame.objects.filter(tagid=tag).values_list('warname', flat=True))
        return a

# a = mysqlOperator().queryWarData()
# mysqlOperator().staticGraph()
# print(a)

# print(mysqlOperator().queryNewTagStaticData('AAA096'))
# print(mysqlOperator().queryTag())
# print(mysqlOperator().FuzzyQueryNewTagStaticData('AAA'))

