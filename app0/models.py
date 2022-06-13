from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


# python manage.py makemigrations
# python manage.py migrate  #--fake

class Frame(models.Model):
    """初始帧表"""
    tagid = models.TextField(max_length=65000)
    warname = models.TextField(max_length=65000)
    # 场景
    SencesNum = models.IntegerField(verbose_name="场景")
    # name
    name = models.TextField(max_length=65000,verbose_name="名称",default='')
    # isRed
    isRed = models.TextField(max_length=65000,verbose_name="红蓝方",default='')
    # type
    type = models.TextField(max_length=65000,default='')
    # value
    value = models.TextField(max_length=65000,default='')
    # ra_Pro_Angle
    ra_Pro_Angle = models.TextField(max_length=65000,default='')
    # ra_Pro_Radius
    ra_Probe_Radius = models.TextField(max_length=65000,default='')
    # ra_StartUp_Delay
    ra_StartUp_Delay = models.TextField(max_length=65000,default='')
    # ra_Process_Delay
    ra_Process_Delay = models.TextField(max_length=65000,default='')
    # ra_Rang_Accuracy
    ra_Rang_Accuracy = models.TextField(max_length=65000,default='')
    # ra_Angle_Accuracy
    ra_Angle_Accuracy = models.TextField(max_length=65000,default='')
    # MisMaxAngle
    MisMaxAngle = models.TextField(max_length=65000,default='')
    # MisMinDisescapeDis
    MisMinDisescapeDis = models.TextField(max_length=65000,default='')
    # MisMaxDisescapeDis
    MisMaxDisescapeDis = models.TextField(max_length=65000,default='')
    # MisMaxV
    MisMaxV = models.TextField(max_length=65000,default='')
    # MisMaxOver
    MisMaxOver = models.TextField(max_length=65000,default='')
    # MisLockTime
    MisLockTime = models.TextField(max_length=65000,default='')
    # MisMinAtkDis
    MisMinAtkDis = models.TextField(max_length=65000,default='')
    MisMaxRange = models.TextField(max_length=65000,default='')
    MisHitPro = models.TextField(max_length=65000,default='')
    ra_Detect_Delay = models.TextField(max_length=65000,default='')
    ra_FindTar_Delay = models.TextField(max_length=65000,default='')
    MisNum = models.TextField(max_length=65000,default='')
    # EchoInitState
    EchoInitState = models.TextField(max_length=65000,default='')
    # EchoFackTarNum
    EchoFackTarNum = models.TextField(max_length=65000,default='')
    # EchoDis
    EchoDis = models.TextField(max_length=65000,default='')
    # SupInitState
    SupInitState = models.TextField(max_length=65000,default='')
    # SupTarNum
    SupTarNum = models.TextField(max_length=65000,default='')
    # SupMinDis
    SupMinDis = models.TextField(max_length=65000,default='')
    SupMaxAngle = models.TextField(max_length=65000,default='')
    planeNum = models.IntegerField(verbose_name="平台数量")

    class Meta:
        managed = True
        db_table = 'Frame'


class Index(models.Model):
    tagid = models.TextField(max_length=65000)
    warname = models.TextField(max_length=65000)
    aveDistanceFirstFindBlue = models.FloatField()
    aveDistanceFirstFindRed = models.FloatField()
    blueMisHitRate = models.FloatField()
    disWhenAt = models.TextField(max_length=65000)
    distanceBeforLock = models.TextField(max_length=65000)
    distanceBeforAt = models.TextField(max_length=65000)
    distanceBeforSe = models.TextField(max_length=65000)
    distanceBeforEi = models.TextField(max_length=65000)
    findDistance = models.TextField(max_length=65000)
    findTime = models.TextField(max_length=65000)
    rateOfSucEarlyWaring = models.TextField(max_length=65000)
    resumeTimeAt = models.TextField(max_length=65000)
    resumeTimeDtAbi = models.TextField(max_length=65000)
    resumeTimeDtAct = models.TextField(max_length=65000)
    resumeTimeDtPropor = models.TextField(max_length=65000)
    resumeTimeEi = models.TextField(max_length=65000)
    resumeTimeFake = models.TextField(max_length=65000)
    resumeTimeLock = models.TextField(max_length=65000)
    resumeTimeShortestEi = models.TextField(max_length=65000)
    resumeTimeShortestFake = models.TextField(max_length=65000)
    resumeTimeShortestLock = models.TextField(max_length=65000)
    tarLockSta = models.TextField(max_length=65000)
    timeBeforLock = models.TextField(max_length=65000)
    timeBeforSe = models.TextField(max_length=65000)
    shape = models.FloatField()
    shapeCoeff = models.FloatField()
    redMisHitRate = models.FloatField()
    resumeTimeDt = models.TextField(max_length=65000)
    timeBeforEi = models.TextField(max_length=65000)
    timeBeforAt = models.TextField(max_length=65000)
    tarDetSta = models.TextField(max_length=65000)
    frameNum = models.IntegerField()

    class Meta:
        managed = True
        db_table = 'Index'



class Admin(models.Model):
    """管理员"""
    username = models.CharField(verbose_name="用户名", max_length=32)
    password = models.CharField(verbose_name="密码", max_length=64)
    privilege = models.IntegerField(verbose_name="权限", default=0)
    nickname = models.CharField(verbose_name="昵称", max_length=64, default='n')


#
class War(models.Model):
    """作战表"""
    # 编号
    number = models.TextField(max_length=65000)
    # 类型
    type = models.IntegerField(verbose_name="作战类型", default=1,
                               validators=[MaxValueValidator(100), MinValueValidator(1)])
    # 总帧数
    frames = models.IntegerField(verbose_name="总帧数", default=1, validators=[MinValueValidator(1)])

    # 其他待定？

    def __str__(self):
        return self.number


class Frame1(models.Model):
    """初始帧表"""
    # 场景
    sences = models.IntegerField(verbose_name="场景")
    # # 帧数id    # 外键
    war = models.ForeignKey(verbose_name="作战id", to="War", to_field="id", on_delete=models.CASCADE)
    # name
    name = models.TextField(max_length=65000,verbose_name="名称",default='')
    # isRed
    isRed = models.TextField(max_length=65000,verbose_name="红蓝方",default='')
    # type
    type = models.TextField(max_length=65000,default='')
    # value
    value = models.TextField(max_length=65000,default='')
    # ra_Pro_Angle
    ra_Pro_Angle = models.TextField(max_length=65000,default='')
    # ra_Pro_Radius
    ra_Pro_Radius = models.TextField(max_length=65000,default='')
    # ra_StartUp_Delay
    ra_StartUp_Delay = models.TextField(max_length=65000,default='')
    # ra_Process_Delay
    ra_Process_Delay = models.TextField(max_length=65000,default='')
    # ra_Rang_Accuracy
    ra_Rang_Accuracy = models.TextField(max_length=65000,default='')
    # ra_Angle_Accuracy
    ra_Angle_Accuracy = models.TextField(max_length=65000,default='')
    # MisMaxAngle
    MisMaxAngle = models.TextField(max_length=65000,default='')
    # MisMinDisescapeDis
    MisMinDisescapeDis = models.TextField(max_length=65000,default='')
    # MisMaxDisescapeDis
    MisMaxDisescapeDis = models.TextField(max_length=65000,default='')
    # MisMaxV
    MisMaxV = models.TextField(max_length=65000,default='')
    # MisMaxOver
    MisMaxOver = models.TextField(max_length=65000,default='')
    # MisLockTime
    MisLockTime = models.TextField(max_length=65000,default='')
    # MisMinAtkDis
    MisMinAtkDis = models.TextField(max_length=65000,default='')
    # EchoInitState
    EchoInitState = models.TextField(max_length=65000,default='')
    # EchoFackTarNum
    EchoFackTarNum = models.TextField(max_length=65000,default='')
    # EchoDis
    EchoDis = models.TextField(max_length=65000,default='')
    # SupInitState
    SupInitState = models.TextField(max_length=65000,default='')
    # SupTarNum
    SupTarNum = models.TextField(max_length=65000,default='')
    # SupMinDis
    SupMinDis = models.TextField(max_length=65000,default='')

class InitFrame(models.Model):
    """初始帧表"""
    # 场景
    SencesNum = models.IntegerField(verbose_name="场景")
    # name
    name = models.TextField(max_length=65000,verbose_name="名称",default='')
    # isRed
    isRed = models.TextField(max_length=65000,verbose_name="红蓝方",default='')
    # type
    type = models.TextField(max_length=65000,default='')
    # value
    value = models.TextField(max_length=65000,default='')
    # ra_Pro_Angle
    ra_Pro_Angle = models.TextField(max_length=65000,default='')
    # ra_Pro_Radius
    ra_Probe_Radius = models.TextField(max_length=65000,default='')
    # ra_StartUp_Delay
    ra_StartUp_Delay = models.TextField(max_length=65000,default='')
    # ra_Process_Delay
    ra_Process_Delay = models.TextField(max_length=65000,default='')
    # ra_Rang_Accuracy
    ra_Rang_Accuracy = models.TextField(max_length=65000,default='')
    # ra_Angle_Accuracy
    ra_Angle_Accuracy = models.TextField(max_length=65000,default='')
    # MisMaxAngle
    MisMaxAngle = models.TextField(max_length=65000,default='')
    # MisMinDisescapeDis
    MisMinDisescapeDis = models.TextField(max_length=65000,default='')
    # MisMaxDisescapeDis
    MisMaxDisescapeDis = models.TextField(max_length=65000,default='')
    # MisMaxV
    MisMaxV = models.TextField(max_length=65000,default='')
    # MisMaxOver
    MisMaxOver = models.TextField(max_length=65000,default='')
    # MisLockTime
    MisLockTime = models.TextField(max_length=65000,default='')
    # MisMinAtkDis
    MisMinAtkDis = models.TextField(max_length=65000,default='')
    # EchoInitState
    EchoInitState = models.TextField(max_length=65000,default='')
    # EchoFackTarNum
    EchoFackTarNum = models.TextField(max_length=65000,default='')
    # EchoDis
    EchoDis = models.TextField(max_length=65000,default='')
    # SupInitState
    SupInitState = models.TextField(max_length=65000,default='')
    # SupTarNum
    SupTarNum = models.TextField(max_length=65000,default='')
    # SupMinDis
    SupMinDis = models.TextField(max_length=65000,default='')
    #  帧数id    # 外键
    tagid = models.ForeignKey(verbose_name="作战id", to="War", to_field="id", on_delete=models.CASCADE)

    class Meta:
        managed = True
        db_table = 'InitFrame'

# class Department(models.Model):
#     """部门表"""
#     title = models.CharField(verbose_name='标题', max_length=32)
#
#     # 格式化  将返回的结果自定义
#     def __str__(self):
#         return self.title
#
#
# class UserInfo(models.Model):
#     """员工表"""
#     name = models.CharField(verbose_name="姓名", max_length=16)
#     password = models.CharField(verbose_name="密码", max_length=64)
#     age = models.IntegerField(verbose_name="年龄")
#     account = models.DecimalField(
#         verbose_name="账户余额", max_digits=10, decimal_places=2, default=0)
#     # create_time = models.DateTimeField(verbose_name="入职时间")
#     create_time = models.DateField(verbose_name="入职时间")  # 只包含年月日，没有时分秒
#     # 级联删除
#     depart = models.ForeignKey(
#         verbose_name="部门", to="Department", to_field="id", on_delete=models.CASCADE)
#     # 置空
#     # depart = models.ForeignKey(to="Department", to_field="id", null=True, blank=True, on_delete=models.SET_NULL)
#
#     # 在Djangp中做的约束，数据库中性别的值只能为1和2，创建了对应关系
#     gender_choices = (
#         (1, "男"),
#         (2, "女"),
#     )
#     gender = models.SmallIntegerField(
#         verbose_name="性别", choices=gender_choices)


class Static_war_data(models.Model):
    war_name = models.CharField(max_length=255, primary_key=True)
    # war_name = models.CharField(max_length=255)
    version = models.IntegerField()
    posmode = models.IntegerField()
    platforms = models.IntegerField()
    comm = models.TextField(max_length=65000)

    class Meta:
        managed = True
        db_table = 'static_war_data'


# 用户表
class UserModel(models.Model):
    user_id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=10, verbose_name='用户名')
    password = models.CharField(max_length=10, verbose_name='密码')
    privilege = models.IntegerField(verbose_name='权限')
    nickname = models.CharField(max_length=255, verbose_name='昵称')

    class Meta:
        db_table = 'user'


class Static_platform_data(models.Model):
    # id = models.IntegerField()
    # war_name = models.ForeignKey('Static_war_data',on_delete=models.CASCADE,db_column='war_name')
    war_name = models.CharField(max_length=255)
    uav_id = models.CharField(max_length=255)  #
    svv = models.BooleanField()
    type = models.IntegerField()
    master = models.BooleanField()
    radar = models.BooleanField()
    fram = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    frdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    sdelay = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    pdelay = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmam = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmdkmax = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmdkmin = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    lng = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    lat = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    height = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    posx = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    posy = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    posz = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    v = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    northv = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    upv = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    eastv = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    gv = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    iv = models.DecimalField(max_digits=30, default=0, decimal_places=10)

    class Meta:
        managed = True
        db_table = 'static_platform_data'
        # unique_together = (('war_name', 'uav_id'),)


class fluxdb_data(models.Model):
    frame = models.IntegerField()
    svv = models.BooleanField()
    master = models.BooleanField()
    radar = models.BooleanField()
    posx = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    posy = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    posz = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    v = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    northv = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    upv = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    eastv = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    psi = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    gv = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    iv = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    comm = models.TextField(max_length=65000)

    class Meta:
        managed = True
        db_table = 'fluxdb_data'


class InitialFrame(models.Model):
    posmode = models.IntegerField()
    SencesNum = models.IntegerField()
    name = models.CharField(max_length=255)
    type = models.IntegerField()
    fram = models.DecimalField(max_digits=30, default=0, decimal_places=15)
    frdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    sdelay = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    pdelay = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    diff = models.IntegerField()
    fmam = models.DecimalField(max_digits=30, default=0, decimal_places=15)
    fmdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmdkmax = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmdkmin = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    isRed = models.IntegerField()
    echo = models.IntegerField()
    fsjtn = models.IntegerField()
    fetn = models.IntegerField()
    fmv = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmload = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    ftdt = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fttp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fsuppress = models.IntegerField()
    fsjdmin = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fdjdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmtlt = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmcp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmhp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    value = models.IntegerField()

    class Meta:
        managed = True
        db_table = 'InitialFrame'


## 探测 init
class Init_1207(models.Model):
    tagid = models.TextField(max_length=65000)
    posmode = models.IntegerField()
    SencesNum = models.IntegerField()
    name = models.CharField(max_length=255)
    type = models.IntegerField()
    fram = models.DecimalField(max_digits=30, default=0, decimal_places=15)
    frdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmam = models.DecimalField(max_digits=30, default=0, decimal_places=15)
    fmdm = models.IntegerField()
    isRed = models.IntegerField()
    fsjtn = models.IntegerField()
    fetn = models.IntegerField()
    fttp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fsjdmin = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fdjdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmcp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmhp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmhdmin = models.FloatField()

    class Meta:
        managed = True
        db_table = 'Init_1207'


## 探测last
class Last_1207(models.Model):
    tagid = models.TextField(max_length=65000)
    SENODE2ofEdge = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    SENODEofEdge = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    aveDistance1 = models.TextField(max_length=65000)
    averageLev001 = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    averageLev002 = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    averageLev003 = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    averageLev004 = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    averageLev2 = models.FloatField()
    betweenNess3 = models.FloatField()
    betweenNess4 = models.FloatField()
    distance1 = models.TextField(max_length=65000)
    fram = models.DecimalField(max_digits=30, default=0, decimal_places=15)
    frdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fttp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    isSurvey = models.BooleanField()
    maxProsum = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    mbsbzql = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    mbtcxxffblhfsj = models.TextField(max_length=65000)
    mbtcxxffzjhfsj = models.TextField(max_length=65000)
    probedNum = models.IntegerField()
    shortest_SeXW_PathLen_Min = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    shortest_SeXW_pre_Min = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    shortest_SeYS_PathLen_Min = models.IntegerField()
    shortest_SeYS_pre_Min = models.IntegerField()
    tchfsj = models.TextField(max_length=65000)
    xdfxjl = models.TextField(max_length=65000)
    xdfxsj = models.TextField(max_length=65000)
    yjcgl = models.DecimalField(max_digits=30, default=0, decimal_places=10)

    class Meta:
        managed = True
        db_table = 'Last_1207'


## 打击 init
class AttackInit(models.Model):
    tagid = models.TextField(max_length=65000)
    posmode = models.IntegerField()
    SencesNum = models.IntegerField()
    name = models.CharField(max_length=255)
    type = models.IntegerField()
    fram = models.DecimalField(max_digits=30, default=0, decimal_places=15)
    frdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmam = models.DecimalField(max_digits=30, default=0, decimal_places=15)
    fmdm = models.IntegerField()
    isRed = models.IntegerField()
    fsjtn = models.IntegerField()
    fetn = models.IntegerField()
    fmv = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmload = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    ftdt = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fttp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fsjdmin = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fdjdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmtlt = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmcp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmhp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    missile_num = models.IntegerField()
    fmhdmin = models.FloatField()

    class Meta:
        managed = True
        db_table = 'AttackInit'


## 打击 last
class AttackLast(models.Model):
    tagid = models.TextField(max_length=65000)
    AttackDis = models.TextField(max_length=65000)
    SENODE2ofEdge = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    SENODEofEdge = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    aveDistance1 = models.TextField(max_length=65000)
    averageLev001 = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    averageLev002 = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    averageLev003 = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    averageLev004 = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    averageLev2 = models.FloatField()
    betweenNess3 = models.FloatField()
    betweenNess4 = models.FloatField()
    distance1 = models.TextField(max_length=65000)
    fmam = models.DecimalField(max_digits=30, default=0, decimal_places=15)
    fmdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fram = models.DecimalField(max_digits=30, default=0, decimal_places=15)
    frdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fttp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    isSurvey = models.BooleanField()
    maxAtcsum = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    maxProsum = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    mbjhxxffzdlj = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    mbjhxxffzdljhfsj = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    mbsbzql = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    mbtcxxffblhfsj = models.TextField(max_length=65000)
    mbtcxxffzjhfsj = models.TextField(max_length=65000)
    missile_num = models.IntegerField()
    probedNum = models.IntegerField()
    shortestAtXWMzxVal = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    shortestAtYSMzxVal = models.IntegerField()
    shortest_SeXW_pre_Min = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    shortest_SeYS_pre_Min = models.IntegerField()
    tchfsj = models.TextField(max_length=65000)
    xdfsjl = models.TextField(max_length=65000)
    xdfssj = models.TextField(max_length=65000)
    xdfxjl = models.TextField(max_length=65000)
    xdfxsj = models.TextField(max_length=65000)
    yjcgl = models.DecimalField(max_digits=30, default=0, decimal_places=10)

    class Meta:
        managed = True
        db_table = 'AttackLast'


# 压制静态
class RepressInit(models.Model):
    tagid = models.TextField(max_length=65000)
    posmode = models.IntegerField()
    SencesNum = models.IntegerField()
    name = models.CharField(max_length=255)
    type = models.IntegerField()
    fram = models.DecimalField(max_digits=30, default=0, decimal_places=15)
    frdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmam = models.DecimalField(max_digits=30, default=0, decimal_places=15)
    fmdm = models.IntegerField()
    isRed = models.IntegerField()
    fsjtn = models.IntegerField()
    fetn = models.IntegerField()
    fmv = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmload = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    ftdt = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fttp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fsjdmin = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fdjdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmtlt = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmcp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmhp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    missile_num = models.IntegerField()
    fmhdmin = models.FloatField()

    class Meta:
        managed = True
        db_table = 'RepressInit'


class RepressLast(models.Model):
    tagid = models.TextField(max_length=65000)
    AttackDis = models.TextField(max_length=65000)
    SENODE2ofEdge = models.FloatField()
    SENODEofEdge = models.FloatField()
    aveDistance1 = models.TextField(max_length=65000)
    averageLev001 = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    averageLev002 = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    averageLev003 = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    averageLev004 = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    averageLev2 = models.FloatField()
    betweenNess3 = models.FloatField()
    betweenNess4 = models.FloatField()
    distance1 = models.TextField(max_length=65000)
    fetn = models.IntegerField()
    fmam = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fmdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fram = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    frdm = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    fttp = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    isSurvey = models.BooleanField()
    m_echoNum = models.IntegerField()
    maxEicsum = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    maxProsum = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    maxYPcsum = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    mbjhxxffzdlj = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    mbjhxxffzdljhfsj = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    mbsbzql = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    mbtcxxffblhfsj = models.TextField(max_length=65000)
    mbtcxxffzjhfsj = models.TextField(max_length=65000)
    missile_num = models.IntegerField()
    probedNum = models.IntegerField()
    shortestAtXWMzxVal = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    shortestEIXWMzxVal = models.IntegerField()
    shortest_SeXW_pre_Min = models.DecimalField(max_digits=30, default=0, decimal_places=10)
    shortest_SeYS_pre_Min = models.IntegerField()
    tchfsj = models.TextField(max_length=65000)
    xdfsjl = models.TextField(max_length=65000)
    xdfssj = models.TextField(max_length=65000)
    xdfxjl = models.TextField(max_length=65000)
    xdfxsj = models.TextField(max_length=65000)
    yjcgl = models.DecimalField(max_digits=30, default=0, decimal_places=10)

    class Meta:
        managed = True
        db_table = 'RepressLast'

