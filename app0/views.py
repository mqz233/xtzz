import json
import os
import shutil
import zipfile
import py7zr
import django_tables2 as tables
from django.contrib import messages
from django.db import models
from django.http import JsonResponse
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.views.decorators.csrf import csrf_exempt
from django_tables2 import RequestConfig
from DataOperator.communityData import community_data_op
from DataOperator.createDataSet import Warwinner
from DataOperator.fluxdbOperator import fluxdbOperator
from DataOperator.influxdb_data_store import dynamic_store, all_plane_store, all_index_store
from DataOperator.mysqlOperator import mysqlOperator as sdo
from DataOperator.newStaticStore import all_static_store, all_search_tag_static_store
from DataOperator.scanAndImportFiles import scanner, stopScanner
from DataOperator.sensitivity_analysis import sensitivity_analysis as sa
import DataOperator.jsonOperator as jo
from HGP.HGPCLASS import HGP
from HGP.prediction import MTMPRE
from app0.models import UserModel, Static_war_data, Static_platform_data, InitialFrame,War,InitFrame,Index
from app0 import models
from app0.utils.pagination import Pagination
# from PredictionModel.trainModel import predictionSvv, readPredictedData
from app0.utils.pagination1 import Pagination1
from app0.utils.pagination2 import Pagination2
from app0.utils.pagination3 import Pagination3
from conf.readConfig import readConfig
from DataOperator.influxdb_data_store import editpos
from HGP.pos_pre import POSPRE

def static_data_store(path,tag):

    # 某一场作战路径
    # path = '../output/'

    # 从index.json读取总帧数,构建作战表
    index_path = path + "index.json"
    frames = jo.jsonOperator().convertJsonToDict(index_path)['frameNum']

    # 从init文件读取初始帧信息
    init_path = ''
    for file in os.listdir(path):
        if file.startswith('init'):
            init_path = path + os.path.basename(file)

    # 存储初始帧信息
    data_content_dict = jo.jsonOperator().convertJsonToDict(init_path)
    # key列表
    keys_list = list(data_content_dict.keys())
    # value列表
    values_list = list(data_content_dict.values())
    # key长度
    len_cols = len(keys_list)

    #生成或者查找作战实例
    if(War.objects.get(number=tag)):
        war_instance = War.objects.get(number='1234')
    else:
        war_instance = War(number=tag,type=1,frames=frames)

    InitFrame(
        SencesNum=data_content_dict['SencesNum'],
        name=data_content_dict['name'],
        isRed=data_content_dict['type'],
        type=data_content_dict['type'],
        value=data_content_dict['value'],
        ra_Pro_Angle=data_content_dict['ra_Pro_Angle'],
        ra_Probe_Radius=data_content_dict['ra_Probe_Radius'],
        ra_StartUp_Delay=data_content_dict['ra_StartUp_Delay'],
        ra_Process_Delay=data_content_dict['ra_Process_Delay'],
        ra_Rang_Accuracy=data_content_dict['ra_Rang_Accuracy'],
        ra_Angle_Accuracy=data_content_dict['ra_Angle_Accuracy'],
        MisMaxAngle=data_content_dict['MisMaxAngle'],
        MisMinDisescapeDis=data_content_dict['MisMinDisescapeDis'],
        MisMaxDisescapeDis=data_content_dict['MisMaxDisescapeDis'],
        MisMaxV=data_content_dict['MisMaxV'],
        MisMaxOver=data_content_dict['MisMaxOver'],
        MisLockTime=data_content_dict['MisLockTime'],
        MisMinAtkDis=data_content_dict['MisMinAtkDis'],
        EchoInitState=data_content_dict['EchoInitState'],
        EchoFackTarNum=data_content_dict['EchoFackTarNum'],
        EchoDis=data_content_dict['EchoDis'],
        SupInitState=data_content_dict['SupInitState'],
        SupTarNum=data_content_dict['SupTarNum'],
        SupMinDis=data_content_dict['SupMinDis'],
        tagid=war_instance,
    ).save()


def del_file(path_data):
    """
        删除文件夹下所有文件
    :param path_data:   文件夹路径，绝对路径
    :return:
    """
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data):  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)


def index(request):
    return render(request, 'index.html')


@csrf_exempt
@xframe_options_sameorigin
def back_test(request):
    return render(request, 'demo.html')


class Static_war_data_Table(tables.Table):
    class Meta:
        model = Static_war_data
        attrs = {'class': 'table table-bordered',  # 'layui-table',
                 }
        template_name = "django_tables2/bootstrap4.html"


class InitialFrame_Table(tables.Table):
    class Meta:
        model = InitialFrame
        attrs = {'class': 'table table-bordered',  # 'layui-table',
                 }
        template_name = "django_tables2/bootstrap4.html"


class Static_platform_data_Table(tables.Table):
    class Meta:
        model = Static_platform_data
        attrs = {'class': 'table table-bordered',  # 'layui-table',
                 }
        template_name = "django_tables2/bootstrap4.html"


class fluxdb_data_table(tables.Table):
    frame = tables.Column(orderable=False)
    svv = tables.Column(orderable=False)
    master = tables.Column(orderable=False)
    radar = tables.Column(orderable=False)
    posx = tables.Column(orderable=False)
    posy = tables.Column(orderable=False)
    posz = tables.Column(orderable=False)
    v = tables.Column(orderable=False)
    northv = tables.Column(orderable=False)
    upv = tables.Column(orderable=False)
    eastv = tables.Column(orderable=False)
    psi = tables.Column(orderable=False)
    gv = tables.Column(orderable=False)
    iv = tables.Column(orderable=False)
    comm = tables.Column(orderable=False)

    class Meta:
        # models = fluxdb_data
        attrs = {'class': 'table table-bordered',  # 'layui-table',
                 }
        template_name = "django_tables2/bootstrap4.html"


# 测试table
@csrf_exempt
@xframe_options_sameorigin
def form_test(request):
    return render(request, 'demo.html')


@csrf_exempt
def upload_page(request):
    return render(request, 'upload.html')


# 上传文件
@csrf_exempt
@xframe_options_sameorigin
def upload_test(request):
    if request.method == 'POST':
        files = request.FILES.getlist('file')
        changci1 = request.POST.get('changci1')
        print(changci1)
        print(files)
        # 上传文件到服务器
        if not files:
            return render(request, 'upload.html', {})
        else:
            for file in files:
                des = open(os.path.join(readConfig().getStaticRootPath(), file.name), 'wb+')

                for chunk in file.chunks():
                    des.write(chunk)
                des.close()
            # 导入数据库
            all_static_store(readConfig().getStaticRootPath())
            # sdo().insertSingleDataFromJson(war_name=changci1, warPath=r'E:/项目数据集/transmission place/R/')
            # 处理过数据移除
        return render(request, 'upload.html', {'context': '已完成'})


# 上传动态文件
@csrf_exempt
@xframe_options_sameorigin
def upload_test2(request):
    if request.method == 'POST':
        files = request.FILES.getlist('file')
        changci2 = request.POST.get('changci2')
        print(changci2)
        print(files)
        path = os.path.join(readConfig().getDynamicRootPath(), changci2)
        folder = os.path.exists(path)

        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
            print("---  new folder...  ---")
            print("---  OK  ---")

        else:
            print("---  There is this folder!  ---")

        if not files:
            return render(request, 'upload.html', {})
        else:
            for file in files:
                des = open(os.path.join(path, file.name), 'wb+')
                for chunk in file.chunks():
                    des.write(chunk)
                des.close()
            dynamic_store(changci2)
            # Dynamic(r'E:/项目数据集/transmission place/push/').store_data(war_name=changci2)
        return render(request, 'upload.html', {'context': '已完成'})


# # 搜索出表
# @csrf_exempt
# @xframe_options_sameorigin
# def search(request):
#     # tag
#     tag = request.POST.get('search_tag')
#
#     start = request.POST.get('start')
#     end = request.POST.get('end')
#     war_name_list = fluxdbOperator().get_measurements()
#     num_battle = request.POST.get('num_battle')
#     print(num_battle)
#     # table = QuerySet.none()
#
#     if not num_battle:
#         table1 = InitialFrame_Table(InitialFrame.objects.all(), prefix="1-")
#         # table2 = Static_platform_data_Table(Static_platform_data.objects.all(), prefix="2-")
#         table3 = fluxdb_data_table([])
#         RequestConfig(request, paginate={"per_page": 10}).configure(table1)
#         # RequestConfig(request, paginate={"per_page": 10}).configure(table2)
#         # RequestConfig(request, paginate={"per_page": 10}).configure(table3)
#         return render(request, 'demo.html', {'table1': table1, 'table3': table3, 'war_name_list': war_name_list})
#     requiredData = []
#     requiredData.append(num_battle)
#     print(num_battle)
#     n_battle = num_battle.split('_')
#     # data1, data2 = sdo().queryStaticData(requiredData)
#     data1 = sdo().queryNewStaticData(n_battle[-1])
#     data3 = fluxdbOperator().select_num_battle(requiredData[0])
#     print(data1)
#     table1 = InitialFrame_Table(data1['initialFrame'])
#
#     # table2 = Static_platform_data_Table(data2)
#     table3 = fluxdb_data_table(list(data3))
#     # table.paginate(page=request.GET.get("page", 1), per_page=10)
#     # RequestConfig(request, paginate={"per_page": 1}).configure(table1)
#
#     # RequestConfig(request, paginate={"per_page": 10}).configure(table2)
#     RequestConfig(request, paginate={"per_page": 10}).configure(table1)
#     # RequestConfig(request, paginate={"per_page": 10}).configure(table2)
#     RequestConfig(request, paginate={"per_page": 10}).configure(table3)
#     return render(request, 'demo.html', {'table1': table1, 'table3': table3, 'war_name_list': war_name_list})


@csrf_exempt
def predict_formal(request):
    tagid = fluxdbOperator().get_plane_measurements()
    sel_tagid = request.POST.get("tagid")
    sel_model = request.POST.get("model")
    if (request.POST.get("epochs")):
        epochs = (int)(request.POST.get("epochs"))
        lr = (float)(request.POST.get("lr"))
        lookback = (int)(request.POST.get("lookback"))
    # print(sel_tagid)
    # get请求，展示基础页面
    if not (sel_tagid):
        context = {
            "model":["LSTM","GRU","Prophet"],
            "tagid": tagid,
            "x1": [],
            "x2": [],
            "x3": [],
            "red1": [],
            "red2": [],
            "red3": [],
            "red4": [],
            "blue1": [],
            "blue2": [],
            "blue3": [],
            "blue4": [],
            "hist": [],
        }
        return render(request, "predict2.html", context)
    else:
        measurement = str(sel_tagid)
        #     创建数据集
        from PredictionModel.LSTM import PRELSTM
        red1, red2, red3, red4, blue1, blue2, blue3, blue4, hist = PRELSTM(measurement,epochs=epochs, lookback=lookback, lr=lr).PRELSTM()
        x1 = [i+20 for i in range(len(red1))]
        x2 = [i+20 for i in range(len(blue1))]
        x3 = [i+1 for i in range(len(hist))]
        context = {
            "sel_tagid":sel_tagid,
            "sel_model":sel_model,
            "tagid": tagid,
            "model": ["LSTM", "GRU", "Prophet"],
            "x1":x1,
            "x2":x2,
            "x3":x3,
            "red1":red1,
            "red2":red2,
            "red3":red3,
            "red4":red4,
            "blue1":blue1,
            "blue2":blue2,
            "blue3":blue3,
            "blue4":blue4,
            "hist":hist,

        }
        return render(request, "predict2.html",context)


@csrf_exempt
def predict(request):
    num_battle = request.POST.get("x")
    predict_method = request.POST.get("y")
    print(num_battle, predict_method)
    # predictionSvv(num_battle, predict_method)
    # return render(request, "predict.html", {'project': project_data})


@csrf_exempt
def headpage(request):
    return render(request, "headpage.html")


@csrf_exempt
def analyse(request):
    x_1 = ''
    x_2 = ''
    x_3 = ''
    x_4 = ''
    x_5 = ''
    x_6 = ''

    switch = {'1': '飞行速度', '2': '机动过载', '3': '处理时延', '4': '探测节点时延', '5': '雷达最大探测角度', '6': '导弹最大发射角度', '7': '导弹最远攻击距离',
              '8': '导弹不可逃逸最大距离', '9': '导弹不可逃逸最小距离'}
    if request.method == 'POST':
        x_1 = request.POST.get('x_1'),
        x_2 = request.POST.get('x_2'),
        x_3 = request.POST.get('x_3'),
        x_4 = request.POST.get('x_4'),
        x_5 = request.POST.get('x_5'),
        x_6 = request.POST.get('x_6'),
    if x_1 == '':
        return render(request, 'charts_full.html',
                      {'x_1': switch['1'], 'x_2': switch['2'], 'x_3': switch['3'], 'x_4': switch['4'],
                       'x_5': switch['5'], 'x_6': switch['6']})
    x1 = switch[x_1[0]]
    x2 = switch[x_2[0]]
    x3 = switch[x_3[0]]
    x4 = switch[x_4[0]]
    x5 = switch[x_5[0]]
    x6 = switch[x_6[0]]
    print(x1, x2, x3, x4, x5, x6)
    # 数据接口
    p_data = []
    return render(request, 'charts_full.html',
                  {'p_data': p_data, 'x_1': x1, 'x_2': x2, 'x_3': x3, 'x_4': x4, 'x_5': x5, 'x_6': x6})


@csrf_exempt
def system_upload(request):
    keyjobs = request.GET.get("keyjobs")
    print(keyjobs)
    if keyjobs == '1':
        # 启动函数
        print("system_upload_start")
        messages.success(request, "已启动")
        # 加入导入情况
        scanner1 = scanner(
            warPath='E:/项目数据集/transmission place/json_output_5055555/',
            sleep_time=0.5
        )
        scanner1.start()
        return render(request, 'upload.html')
    #     return "system_upload_start"
    #
    if keyjobs == '0':
        # 中止函数
        stopScanner()
        print("system_upload_end")
        messages.success(request, "已中止")
        return render(request, 'upload.html')


@csrf_exempt
def chart_part(request):
    if request.method == 'POST':
        x = request.POST.get('x')
        y = request.POST.get('y')
        print(x, y)
    # x_data = []
    # y_data = []
    # x_name = []
    # y_name = []
    # full_data = []
    # return render(request, 'chart_part.html', {'x_data': x_data, 'y_data': y_data, 'x_name': x_name, 'y_name': y_name, "full_data": full_data})
    return render(request, 'chart_part.html')
    # {'x_data': x_data, 'y_data': y_data, 'x_name': x_name, 'y_name': y_name, "full_data": full_data})


@csrf_exempt
def authority_manage(request):
    do = request.POST.get('do')
    privilege = request.POST.get('privilege')
    username = request.POST.get('username')
    nickname = request.POST.get('nickname')
    password = request.POST.get('password')
    print(do, privilege, username, nickname, password)
    if request.method == "POST":
        if not all([username, password]):
            context = {
                'status': '错误！用户名、密码及名称不能为空！',
                'length': 0
            }
            return render(request, 'authority_manage.html', context)
        else:
            if do == '1':
                user = UserModel.objects.filter(username=username)
                if len(user):
                    context = {
                        'status': '错误！用户名已存在',
                        'length': 0
                    }
                    return render(request, 'authority_manage.html', context)

                else:
                    # 插入
                    UserModel.objects.create(privilege=privilege, username=username, nickname=nickname,
                                             password=password)
                    context = {
                        'status': '创建成功'
                    }
                    return render(request, 'authority_manage.html', context)
            if do == '2':
                user = UserModel.objects.filter(username=username)
                user.update(privilege=privilege, nickname=nickname, password=password)
                context = {
                    'status': '编辑成功'
                }
                return render(request, 'authority_manage.html', context)
            if do == '3':
                user = UserModel.objects.filter(username=username)
                # 删除
                user.delete()
                context = {
                    'status': '删除成功'
                }
                return render(request, 'authority_manage.html', context)

    else:
        return render(request, 'authority_manage.html')


@csrf_exempt
def community_data_mine(request):
    # res = {'a': 1}
    perdict_result_list = [1, 2, 3]
    return render(request, 'community_data_mine.html', {'tag_list': perdict_result_list})


@csrf_exempt
def json_url(request):
    res = request.POST.get('res')
    return JsonResponse(res)


@csrf_exempt
def community_data(request):
    tagid = sdo().queryTag()
    tag_name = request.POST.get("tag_name")
    print(tag_name)
    choice = request.POST.get('x_aris')


    print(choice)
    if (choice):
        res,namedic = community_data_op(choice)
        tag_name = request.POST.get('tag_name')
        # print(namedic)
        # return JsonResponse(res)
        return render(request, 'community_data_mine.html',
                      {"tag_name":tag_name,"tag_list": tagid,'res': res, 'x_name': choice, 'namedic':namedic})
    else:
        res = {}
        return render(request, 'community_data_mine.html',
                      { "tag_list": tagid,})


# @csrf_exempt
# def chart_v2(request):
#     return render(request, 'chart_v2.html')

@csrf_exempt
def chart_v2(request):
    tagid = sdo().queryTag()
    tag_name = request.POST.get("tag_name")
    # get请求，展示基础页面
    if not (tag_name):
        x = []
        y1 = []
        y2 = []
        y3 = []
        y5 = []
        y6 = []
        y7 = []
        y9 = []
        context = {
            "tag_list": tagid,
            'x': x, 'y1': y1, 'y2': y2, 'y3': y3, 'y5': y5, 'y6': y6, 'y7': y7, 'y9': y9,
        }
        return render(request, "chart_v2.html", context)
    else:
        index = request.POST.get("x_aris")
        result = sa().get_graph_data(int(index), str(tag_name))

        x = result['x']
        print(result)
        y1 = result['y'][0]
        y2 = result['y'][1]
        y3 = result['y'][2]

        y5 = result['y'][3]
        y6 = result['y'][4]
        y7 = result['y'][5]

        y9 = result['y'][6]

        context = {
            'x': x[0:10], 'y1': y1[0:10], 'y2': y2[0:10], 'y3': y3[0:10], 'y5': y5[0:10], 'y6': y6[0:10],
            'y7': y7[0:10], 'y9': y9[0:10], 'x_name': index, 'tag_list': tagid,'tag_name': tag_name

        }
        return render(request, "chart_v2.html", context)


@csrf_exempt
def community_mine(request):
    return render(request, 'community_mine.html')

def check_contain_chinese(check_str):
   for ch in check_str.encode('utf-8').decode('utf-8'):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
        return False


@csrf_exempt
def upload_zip(request):
    files = request.FILES.getlist('file')
    # 上传zip时输入的tag
    tag = request.POST.get('zip_tag')
    if not tag:
        return render(request, 'upload.html', {'context': '请输入tag'})
    # print(type(tag.encode()))
    # tagid = str(tag.encode())
    tagid = str(tag)
    if(check_contain_chinese(tagid)):
        return render(request, 'upload.html', {'context': 'tag暂不支持中文'})
    print("tag", tag)
    if files:
        print("shoudao")
        print(files)

    if not files:
        return render(request, 'upload.html', {})
    else:
        # 5.删除zip，unzip两个文件夹下所有文件
        for dir_list in os.listdir(readConfig().getUnZipRootPath()):  # 删除unzip下的所有文件
            shutil.rmtree(os.path.join(readConfig().getUnZipRootPath(), dir_list))
        del_file(readConfig().getZipRootPath())
        # for file in files:
        #     # 1.把压缩文件存到zip文件夹
        #     des = open(os.path.join(readConfig().getZipRootPath(), file.name), 'wb+')
        #     print("shoudao")
        #     for chunk in file.chunks():
        #         des.write(chunk)
        #     des.close()
        #
        #     # 2.解压压缩包
        #     if (file.name.endswith("7z")):
        #         with py7zr.SevenZipFile(os.path.join(readConfig().getZipRootPath(), file.name), mode='r') as z:
        #             z.extractall(path=readConfig().getUnZipRootPath())
        #             # z.close()
        #     else:
        #         zfile = zipfile.ZipFile(os.path.join(readConfig().getZipRootPath(), file.name))
        #         zfile.extractall(path=readConfig().getUnZipRootPath())
        #         zfile.close()
        #
        #     unzip = os.listdir(readConfig().getUnZipRootPath())[0]
        #     unzip = os.path.join(readConfig().getUnZipRootPath(), unzip)
        #     # 3.存动态数据
        #     for dir in os.listdir(unzip):
        #         final_path1 = unzip + '\\' + dir + '\\plane'
        #         final_path2 = unzip + '\\' + dir + '\\index'
        #         all_index_store(tagid, final_path2)
        #         all_plane_store(tagid, final_path1)
        #
        #     # 4.存静态数据
        #     for dir in os.listdir(unzip):
        #         final_path = unzip + '\\' + dir
        #         all_static_store(final_path, tagid)
        #
        #     for dir_list in os.listdir(readConfig().getUnZipRootPath()):  # 删除unzip下的所有文件
        #         shutil.rmtree(os.path.join(readConfig().getUnZipRootPath(), dir_list))
        #     del_file(readConfig().getZipRootPath())
        #     render(request, 'upload.html', {'context': '上传成功'})



        try:
            for file in files:
                # 1.把压缩文件存到zip文件夹
                des = open(os.path.join(readConfig().getZipRootPath(), file.name), 'wb+')
                print("shoudao")
                for chunk in file.chunks():
                    des.write(chunk)
                des.close()

                # 2.解压压缩包
                # print(file.name)
                zfile = zipfile.ZipFile(os.path.join(readConfig().getZipRootPath(), file.name))
                zfile.extractall(path=readConfig().getUnZipRootPath())
                zfile.close()
                unzip = os.listdir(readConfig().getUnZipRootPath())[0]
                unzip = os.path.join(readConfig().getUnZipRootPath(),unzip)

                # 3.存动态数据
                for dir in os.listdir(unzip):
                    final_path1 = unzip+'\\'+dir+'\\plane'
                    final_path2 = unzip+ '\\' + dir + '\\index'
                    all_index_store(tagid,final_path2)
                    all_plane_store(tagid,final_path1)

                # 4.存静态数据
                for dir in os.listdir(unzip):
                    final_path = unzip+'\\'+dir
                    all_static_store(final_path,tagid)

        except Exception as e:
            return render(request, 'upload.html', {'context': '上传失败，请检查压缩包格式'})
        finally:
            # 5.删除zip，unzip两个文件夹下所有文件
            for dir_list in os.listdir(readConfig().getUnZipRootPath()):  # 删除unzip下的所有文件
                shutil.rmtree(os.path.join(readConfig().getUnZipRootPath(), dir_list))
            del_file(readConfig().getZipRootPath())

    return render(request, 'upload.html', {'context': '上传成功'})



@csrf_exempt
def comm_dig(request):
    tagid = sdo().queryTag()
    # tag_name = request.POST.get("tag_name")
    perdict_result_list = [1, 2, 3]
    tag_name = request.POST.get("tag_name")
    print(tag_name)
    choice = request.POST.get('x_aris')
    if (choice != None):
        if (choice == "一类数据挖掘分析"):
            return render(request, 'comm_dig.html', {'choice': 1, 'tag_list': tagid,"tag_name":tag_name})
        elif(choice == "二类数据挖掘分析"):
            return render(request, 'comm_dig.html', {'choice': 2, 'tag_list': tagid,"tag_name":tag_name})
        else:
            return render(request, 'comm_dig.html', {'choice': 3, 'tag_list': tagid, "tag_name": tag_name})

    return render(request, 'comm_dig.html', {'tag_list': tagid})


@csrf_exempt
def error_404(request,exception):
    return render(request, "error_404.html",status=404)

# def return_to_index(request):
#     redirect("/admin_manage")
# 模拟训练函数
def train(epochs, batch_size, lr):
    print(epochs, batch_size, lr)
    reg_list = [1, 1, 1, 2, 2, 3, 3, 1, 1, 2]
    actual_list = [1, 1, 2, 3, 2, 3, 2, 1, 2, 2]
    train_loss = [0.8, 0.85, 0.82, 0.91, 0.95]
    test_loss = [0.4, 0.55, 0.49, 0.54, 0.6]
    return reg_list, actual_list, train_loss, test_loss


# 态势识别
@csrf_exempt
def pos_reg(request):
    # 获取用户选择的数据集和模型
    sel_dataset = request.POST.get("dataset")
    sel_model = request.POST.get("model")
    # 获取用户选择的超参数
    if (request.POST.get("epochs")):
        epochs = (int)(request.POST.get("epochs"))
        lr = (float)(request.POST.get("lr"))
        batch_size = (int)(request.POST.get("batch_size"))
    # 从数据库获取当前的所有tagid
    # dataset = sdo().queryWarData()
    # ds = []
    # for number in dataset:
    #     ds.append(number['number'])
    # get请求，展示基础页面
    if not (sel_dataset):
        return render(request, "pos_reg.html",
                      {'x1': [], 'y1': [], 'y2': [], 'x2': [], 'y3': [], 'y4': [], 'dataset': fluxdbOperator().get_plane_measurements(),
                       'model': ['HGP-SL', 'CapsGNN', 'S2GN']})
    # 调用训练函数，接收网页传来的model和dataset两个参数
    # 返回值是识别列表，实际列表，训练loss，测试loss
    # if (sel_dataset == "ENZYMES"):
    pre_list, real_list, train_loss, valid_loss = HGP(measurement=sel_dataset, epochs=epochs, batch_size=batch_size,
                                                          lr=lr).pos_reg()
    # else:
    #     pre_list, real_list, train_loss, valid_loss = HGP(path="data/MTM-1", epochs=epochs, batch_size=batch_size,
    #                                                       lr=lr).MTM()

    # 图1柱状图x1,y1,y2 图2折线图x2,y3,y4
    x1 = [i for i in range(5)]
    x2 = [i + 1 for i in range(len(train_loss))]
    y1 = pre_list
    y2 = real_list
    y3 = train_loss
    y4 = valid_loss
    # post请求展示训练后的结果
    return render(request, "pos_reg.html",
                  {'frame': [i * 5 for i in range((int)(len(pre_list) / 5))], 'x1': x1, 'y1': y1, 'y2': y2, 'x2': x2,
                   'y3': y3, 'y4': y4, 'dataset': fluxdbOperator().get_plane_measurements(), 'model': ['HGP-SL', 'CapsGNN', 'S2GN'],
                   'sel_dataset': sel_dataset, 'sel_model': sel_model})


# 态势预测
@csrf_exempt
def pos_pre(request):
    # 获取选择的数据集和模型
    sel_dataset = request.POST.get("dataset")
    sel_model = request.POST.get("model")
    if (request.POST.get("epochs")):
        epochs = (int)(request.POST.get("epochs"))
        lr = (float)(request.POST.get("lr"))
        lookback = (int)(request.POST.get("lookback"))
    # get请求，展示基础页面
    if not (sel_dataset):
        return render(request, "pos_pre.html",
                      {'x1': [], 'y1': [], 'y2': [], 'x2': [], 'y3': [], 'dataset': fluxdbOperator().get_plane_measurements(),
                       'model': ['HGP-SL+LSTM', 'CapsGNN+LSTM', 'S2GN+LSTM']})
    # 调用训练函数，接收网页传来的model和dataset两个参数
    # 返回值是识别列表，实际列表，训练loss，测试loss
    pre_list, real_list, train_loss = POSPRE(measurement=sel_dataset,epochs=epochs, lookback=lookback, lr=lr).pospre()
    # 图1柱状图x1,y1,y2 图2折线图x2,y3,y4
    x1 = [i for i in range(len(pre_list))]
    x2 = [i + 1 for i in range(len(train_loss))]
    y1 = pre_list
    y2 = real_list
    y3 = train_loss
    # post请求展示训练后的结果
    return render(request, "pos_pre.html",
                  {'x1': x1, 'y1': y1, 'y2': y2, 'x2': x2, 'y3': y3, 'dataset': fluxdbOperator().get_plane_measurements(),
                   'model': ['HGP-SL+LSTM', 'CapsGNN+LSTM', 'S2GN+LSTM'], 'sel_dataset': sel_dataset,
                   'sel_model': sel_model})

@csrf_exempt
def admin_manage(request):
    """管理员列表"""

    # 搜索
    data_dict = {}
    search_data = request.GET.get('q', "")
    if search_data:
        data_dict["username__contains"] = search_data
        queryset = models.Admin.objects.filter(**data_dict)
    else:
        # 根据搜索条件去数据库获取
        queryset = models.Admin.objects.all(**data_dict)

    # queryset = models.Admin.objects.filter(username=search_data)

    # 分页
    page_object = Pagination(request, queryset)

    context = {
        "queryset": page_object.page_queryset,
        # "queryset": queryset,
        "page_string": page_object.html(),
        "search_data": search_data,
    }

    return render(request, "admin_manage.html", context)


from django import forms
from django.core.exceptions import ValidationError
from app0.utils.bootstrap import BootStrapModelForm


class AdminModelForm(BootStrapModelForm):
    confirm_password = forms.CharField(
        label="确认密码",
        widget=forms.PasswordInput,
    )

    class Meta:
        model = models.Admin
        fields = ["username", "password", "confirm_password", "privilege", "nickname"]
        widgets = {
            "password": forms.PasswordInput(render_value=True),  # 确认密码错了，密码不会清空
        }

    # 对密码进行加密
    def clean_password(self):
        pwd = self.cleaned_data.get("password")
        return md5(pwd)

    # 验证密码和确认密码一致
    def clean_confirm_password(self):
        # print(form.cleaned_data)  验证通过的所有信息
        # 例：{'username':'admin','password':'123,'confirm_password':'3333'}
        pwd = self.cleaned_data.get("password")
        confirm = md5(self.cleaned_data.get("confirm_password"))
        if confirm != pwd:
            raise ValidationError("密码不一致")
        # return什么，保存到数据库的就是什么
        return confirm


def admin_add(request):
    """添加管理员"""
    title = "新建管理员"
    if request.method == "GET":
        form = AdminModelForm()
        return render(request, "change.html", {"form": form, "title": title})

    form = AdminModelForm(data=request.POST)
    if form.is_valid():
        form.save()
        return redirect("/admin_manage")

    return render(request, "change.html", {"form": form, "title": title})


class AdminEditModelForm(BootStrapModelForm):
    class Meta:
        model = models.Admin
        fields = ["username", "privilege", "nickname"]


def admin_edit(request, nid):
    """编辑管理员"""
    # 对象 or None
    row_object = models.Admin.objects.filter(id=nid).first()

    if not row_object:
        return render(request, "error.html", {"msg": "数据不存在"})

    title = "编辑管理员"

    if request.method == "GET":
        form = AdminEditModelForm(instance=row_object)
        return render(request, "change.html", {"form": form, "title": title})

    form = AdminEditModelForm(data=request.POST, instance=row_object)
    if form.is_valid():
        form.save()
        return redirect('/admin_manage')

    return render(request, "change.html", {"form": form, "title": title})


def admin_delete(request, nid):
    """删除管理员"""
    models.Admin.objects.filter(id=nid).delete()
    return redirect('/admin_manage')


class AdminResetModelForm(BootStrapModelForm):
    confirm_password = forms.CharField(
        label="确认密码",
        widget=forms.PasswordInput(render_value=True),
    )

    class Meta:
        model = models.Admin
        fields = ["password", "confirm_password"]
        widgets = {
            "password": forms.PasswordInput(render_value=True),
        }

    def clean_password(self):
        pwd = self.cleaned_data.get("password")
        md5_pwd = md5(pwd)
        # 去数据库校验 当前密码和新密码是否一致
        exists = models.Admin.objects.filter(id=self.instance.pk, password=md5_pwd).exists()
        if exists:
            raise ValidationError("不能与以前的密码相同")

        return md5_pwd

    def clean_confirm_password(self):
        pwd = self.cleaned_data.get("password")
        confirm = md5(self.cleaned_data.get("confirm_password"))
        if confirm != pwd:
            raise ValidationError("密码不一致")
        return confirm


def admin_reset(request, nid):
    """重制密码"""
    row_object = models.Admin.objects.filter(id=nid).first()

    if not row_object:
        return redirect('/admin_manage')

    title = "重置密码 - {}".format(row_object.username)
    if request.method == "GET":
        form = AdminResetModelForm()
        return render(request, "change.html", {"form": form, "title": title})

    form = AdminResetModelForm(data=request.POST, instance=row_object)
    if form.is_valid():
        form.save()
        return redirect('/admin_manage')

    return render(request, "change.html", {"form": form, "title": title})


from django import forms
from django.shortcuts import render, redirect, HttpResponse
from app0 import models
from app0.utils.encrypt import md5
from app0.utils.bootstrap import BootStrapForm
from app0.utils.code import check_code


class LoginForm(BootStrapForm):
    username = forms.CharField(
        label="用户名",
        widget=forms.TextInput,
        required=True,
    )
    password = forms.CharField(
        label="密码",
        widget=forms.PasswordInput,
        required=True,
    )
    code = forms.CharField(
        label="验证码",
        widget=forms.TextInput,
        required=True,
    )

    def clean_password(self):
        pwd = self.cleaned_data.get("password")
        return md5(pwd)


def login(request):
    """登录"""
    if request.method == "GET":
        form = LoginForm()
        return render(request, 'login.html', {"form": form})

    form = LoginForm(data=request.POST)
    if form.is_valid():
        # {"username": 'wupeiqi', "password": '123', "code": ADKFH}
        user_input_code = form.cleaned_data.pop('code')
        code = request.session.get('image_code', "")
        if code.upper() != user_input_code.upper():
            form.add_error("code", "验证码错误")
            return render(request, 'login.html', {"form": form})

        # 去数据库校验用户名和密码是否正确，获取用户对象、None
        admin_object = models.Admin.objects.filter(**form.cleaned_data).first()
        if not admin_object:
            form.add_error("password", "用户名或密码错误")
            return render(request, 'login.html', {"form": form})

        # 用户名和密码正确
        # 网站生成随机字符串；写到用户浏览器的cookies中；再写入到session中
        request.session["info"] = {'id': admin_object.id, 'name': admin_object.username,
                                   'privilege': admin_object.privilege}
        # session可以保存7天 (修改)
        # request.session.set_expiry(60 * 60 * 24 * 7)
        request.session.set_expiry(0)  # 关闭浏览器即失效
        return redirect("/index/")

    return render(request, 'login.html', {"form": form})


from io import BytesIO


def image_code(request):
    """生成图片验证码"""
    # 调用pillow函数，生成图片
    img, code_string = check_code()
    print(code_string)

    # 生成的图片验证码写入到session中，以便后续获取验证码进行校验
    request.session['image_code'] = code_string
    # 给session设置60s超时，60秒后图片验证码无效
    request.session.set_expiry(60)

    stream = BytesIO()
    img.save(stream, 'png')
    return HttpResponse(stream.getvalue())


def logout(request):
    """注销"""
    request.session.clear()
    return redirect('/logout/')

# 动态index展示部分
def war_list(request):

    tagid = fluxdbOperator().get_index_measurements()
    sel_tagid = request.POST.get("tagid")
    print(sel_tagid)
    # get请求，展示基础页面
    if not (sel_tagid):
        context = {
            "tagid":tagid
        }
        return render(request, "war_list.html", context)
    else:
        # post请求，展示选择的tag_id
        # 默认展示的指标
        vals1 = ["FackTarNum", "L_aver_ooda"]
        # vals1 = ["L_aver_ooda", "ThreatActionCoefAt", "ThreatActionCoefCc", "ThreatActionCoefCom",
        #   "ThreatCoefAt","ThreatCoefCc","ThreatCoefCom","ability_SE_scale","actedAdvPre","actedPre",
        #   "detectAdvPre","detectPre","index1","index2","lockAdvPre","lockPre",
        #   "sencesTime","taskScaleB","taskScaleR","underShootPreo","wasteScale"]
        # 根据tag_id搜
        #列名
        names = fluxdbOperator().select_column(sel_tagid)
        width = 'width:' + str(1 / len(vals1)) + '%'
        #每一项
        queryset = fluxdbOperator().select_num_battle(str(sel_tagid))
        page_object = Pagination3(request, queryset)
        context = {
        # "queryset": queryset,  # 取数据
        "queryset": page_object.page_queryset,  # 取数据
        "page_string": page_object.html(),  # 取页码
        "names": names,  # 所有的字段
        "sel_tagid":sel_tagid,
        "tagid": tagid,
        "vals1":vals1,
        "width":width

        }
        return render(request, "war_list.html", context)

# 动态index展示2
def war_list2(request):

    tagid = fluxdbOperator().get_index_measurements()
    sel_tagid = request.POST.get("tagid")
    vals1 = request.POST.getlist('check_box_list')
    # post请求，展示选择的tag_id
    # 根据tag_id搜
    # 列名
    names = fluxdbOperator().select_column(sel_tagid)
    width = 'width:' + str(1 / len(vals1)) + '%'
    # 每一项
    queryset = fluxdbOperator().select_num_battle(str(sel_tagid))
    # page_object = Pagination(request, queryset)
    page_object = Pagination3(request, queryset)
    # if(request.POST.get("page")):
    #         page_object = Pagination1(request, queryset)
    #         print("1")
    # else:
    #         page_object = Pagination2(request, queryset)
    #         print("2")
    context = {
        # "queryset": queryset,  # 取数据
        "queryset": page_object.page_queryset,  # 取数据
        "page_string": page_object.html(),  # 取页码
        "names": names,  # 所有的字段
        "sel_tagid": sel_tagid,
        "tagid": tagid,
        "vals1": vals1,
        "width": width

    }
    return render(request, "war_list.html", context)

# 动态plane展示部分
def plane_list(request):

    tagid = fluxdbOperator().get_plane_measurements()
    sel_tagid = request.POST.get("tagid")
    print(sel_tagid)
    # get请求，展示基础页面
    if not (sel_tagid):
        context = {
            "tagid":tagid
        }
        return render(request, "plane_list.html", context)
    else:
        # post请求，展示选择的tag_id
        # 默认展示的指标
        vals1 = ["sences", "frameId","Time","name","svv"]
        # vals1 = ["sences","frameId","time","name","svv","isRed",
        #   "type","value","ra_Pro_Angle","ra_Pro_Radius","ra_StartUp_Delay","ra_Detect_Delay",
        #   "ra_Process_Delay","ra_FindTar_Delay","ra_Rang_Accuracy","ra_Angle_Accuracy","MisMaxAngle","MisMaxRange",
        #   "MisMinDisescapeDis","MisMaxDisescapeDis","MisMaxV","MisMaxOver","MisLockTime","MisHitPro",
        #   "MisMinAtkDis","EchoInitState","EchoFackTarNum","EchoDis","SupInitState","SupTarNum",
        #   "SupMinDis","posx","posy","posz","V","Vn",
        #   "Vu", "Ve", "yaw", "pitch", "roll", "radar_flag",
        #   "rsuppress_flag", "echo_flag", "targetNum", "radar_radius", "atcNum", "controlNum",
        #   "comNum", "suppressNum", "echoNum", "radarList", "locked", "det_pro",
        #   "range_acc", "angle_acc", "atkList", "conList", "comm", "suppressList",
        #   "echo"]
        # 根据tag_id搜
        #列名
        names = fluxdbOperator().select_column(sel_tagid)
        width = 'width:' + str(1 / len(vals1)) + '%'
        #每一项
        queryset = fluxdbOperator().select_num_battle(str(sel_tagid))
        page_object = Pagination3(request, queryset)
        context = {
        # "queryset": queryset,  # 取数据
        "queryset": page_object.page_queryset,  # 取数据
        "page_string": page_object.html(),  # 取页码
        "names": names,  # 所有的字段
        "sel_tagid":sel_tagid,
        "tagid": tagid,
        "vals1":vals1,
        "width":width

        }
        return render(request, "plane_list.html", context)

# plane展示部分2
def plane_list2(request):

    tagid = fluxdbOperator().get_plane_measurements()
    sel_tagid = request.POST.get("tagid")
    vals1 = request.POST.getlist('check_box_list')
    # post请求，展示选择的tag_id
    # 根据tag_id搜
    # 列名
    names = fluxdbOperator().select_column(sel_tagid)
    width = 'width:' + str(1 / len(vals1)) + '%'
    # 每一项
    queryset = fluxdbOperator().select_num_battle(str(sel_tagid))
    # page_object = Pagination(request, queryset)
    page_object = Pagination3(request, queryset)
    # if(request.POST.get("page")):
    #         page_object = Pagination1(request, queryset)
    #         print("1")
    # else:
    #         page_object = Pagination2(request, queryset)
    #         print("2")
    context = {
        # "queryset": queryset,  # 取数据
        "queryset": page_object.page_queryset,  # 取数据
        "page_string": page_object.html(),  # 取页码
        "names": names,  # 所有的字段
        "sel_tagid": sel_tagid,
        "tagid": tagid,
        "vals1": vals1,
        "width": width

    }
    return render(request, "plane_list.html", context)
from app0.utils.form import FrameModelForm, IndexForm


# 静态初始帧列表
def frame_list(request):

    # 搜索
    data_dict = {}
    search_data = request.POST.get('q', "")
    # 根据选择的指标，对内容进行展示
    vals = request.POST.getlist('check_box_list')
    if vals:
        # 用户选择的指标
        print(vals)
    else:
        # 默认展示的指标
        vals = ["SencesNum", "tagid", "name", "isRed"]

    """单帧管理"""
    width = 'width:' + str(1 / len(vals)) + '%'
    names = models.Frame._meta.get_fields()
    page = request.GET.get("page")
    # queryset = models.Frame.objects.all()
    if search_data:
        # vals = request.POST.getlist('check_box_list')
        data_dict["tagid__contains"] = search_data
        queryset = models.Frame.objects.filter(**data_dict)
    else:
        # 根据搜索条件去数据库获取
        search_data = ""
        queryset = models.Frame.objects.all(**data_dict)
    if(request.POST.get("page")):
            page_object = Pagination1(request, queryset)
            print("1")
    else:
            page_object = Pagination2(request, queryset)
            print("2")



    context = {
        "queryset": page_object.page_queryset,  # 取数据
        "page_string": page_object.html(),  # 取页码
        "names": names,  # 所有的字段
        "vals": vals,
        "width": width,
        "search_data":search_data,
        "page":page
    }
    return render(request, "frame_list.html", context)


def frame_add(request):
    """添加单帧（ModelForm）"""
    if request.method == "GET":
        form = FrameModelForm()
        return render(request, "frame_add.html", {"form": form})

    # 校验数据
    form = FrameModelForm(data=request.POST)
    if form.is_valid():
        # 如果数据合法，保存到数据库
        # print(form.cleaned_data)
        # {'name': '阿德', 'password': '666', 'age': 22, 'account': Decimal('2000'), 'create_time': datetime.datetime(2022, 4, 11, 0, 0, tzinfo=<DstTzInfo 'Asia/Shanghai' CST+8:00:00 STD>), 'gender': 1, 'depart': <Department: IT运维部>}
        form.save()
        return redirect("/frame_list/")

    # 校验失败，在页面中显示错误信息
    # print(form.errors)    form中封装了各种情况的错误信息
    # GET的form啥也没有，POST的form有：用户提交的数据 + 验证失败后内部包含的错误信息
    return render(request, 'frame_add.html', {"form": form})


def frame_edit(request, page,nid):
    """编辑单帧"""

    # 根据ID去数据库获取要编辑的哪一行数据(对象)
    row_object = models.Frame.objects.filter(id=nid).first()

    if not row_object:
        return render(request, "error.html", {"msg": "数据不存在"})

    if request.method == "GET":
        form = FrameModelForm(instance=row_object)
        return render(request, "frame_edit.html", {"form": form,"page":page})

    form = FrameModelForm(data=request.POST, instance=row_object)  # 将用户提交的数据更新到这一行
    if form.is_valid():
        form.save()
        # 默认保存用户输入的值，如果还想保存其他值：
        # form.isinstance.字段名 = 值
        return redirect("/frame_list/?page="+request.POST.get("page"))
    return render(request, "frame_edit.html", {"form": form})


def frame_delete(request, nid):
    models.Frame.objects.filter(id=nid).delete()
    return redirect("/frame_list")


# index列表
def index_list(request):
    # 根据选择的指标，对内容进行展示
    if request.method == "POST":
        # 用户选择的指标
        vals = request.POST.getlist('check_box_list')
        print(vals)
    else:
        # 默认展示的指标
        vals =  ["tagid","frameNum"]
    """单帧管理"""
    width = 'width:'+str(1/len(vals))+'%'
    names = Index._meta.get_fields()
    queryset = Index.objects.all()
    page = request.GET.get("page")

    if(request.POST.get("page")):
            page_object = Pagination1(request, queryset)
            print("1")
    else:
            page_object = Pagination2(request, queryset)
            print("2")
    # page_object = Pagination(request, queryset)
    context = {
        "queryset": page_object.page_queryset,  # 取数据
        "page_string": page_object.html(),  # 取页码
        "names": names,  # 所有的字段
        "vals": vals,
        "width":width,
        "page":page
    }
    return render(request, "index_list.html", context)


def index_edit(request, page,nid):
    """编辑单帧"""

    # 根据ID去数据库获取要编辑的哪一行数据(对象)
    row_object = Index.objects.filter(id=nid).first()

    if not row_object:
        return render(request, "error.html", {"msg": "数据不存在"})

    if request.method == "GET":
        form = IndexForm(instance=row_object)
        return render(request, "index_edit.html", {"form": form,"page":page})

    form = IndexForm(data=request.POST, instance=row_object)  # 将用户提交的数据更新到这一行
    if form.is_valid():
        form.save()
        # 默认保存用户输入的值，如果还想保存其他值：
        # form.isinstance.字段名 = 值
        return redirect("/index_list/?page="+request.POST.get("page"))
    return render(request, "index_edit.html", {"form": form})


def index_delete(request, nid):
    models.Index.objects.filter(id=nid).delete()
    return redirect("/index_list")


def win_info(request):
    tagid = sdo().queryTag()
    sel_tagid = request.POST.get("tagid")
    # get请求，展示基础页面
    if not (sel_tagid):
        context = {
            "tagid": tagid
        }
        return render(request, "win_info.html", context)
    else:
        warname, red, blue, draw = Warwinner(sel_tagid).winner()
        info_list = []
        info_list.append({"a":"红方","b":len(red),"c":red},)
        info_list.append({"a":"蓝方","b":len(blue),"c":blue},)
        info_list.append({"a":"平局","b":len(draw),"c":draw},)
        info_list.append({"a":"总场数","b":len(warname),"c":warname},)
        # post请求，展示选择的tag_id
        # 根据tag_id搜
        # 列名        # 每一项
        context = {
            "sel_tagid": sel_tagid,
            "tagid": tagid,
            "info_list":info_list

        }
        return render(request, "win_info.html", context)

def index_mark(request):
    #取表名
    tagid = fluxdbOperator().get_plane_measurements()
    #取选中的tagid
    sel_tagid = request.POST.get("tagid")
    # get请求，展示基础页面
    if not (sel_tagid):
        context = {
            "tagid": tagid
        }
        return render(request, "index_mark.html", context)
    else:
        # post请求，展示选择的tag_id
        # 默认展示的指标
        vals1 = ["frameId", "stage", "eval"]
        # 根据tag_id搜
        width = 'width:' + str(1 / len(vals1)) + '%'
        # 每一项
        client = fluxdbOperator()
        result = client.select_num_battle(str(sel_tagid))
        queryset = []
        for i in range(0, len(result), 50):
            queryset.append(result[i])
        # queryset = fluxdbOperator().select_num_battle(str(sel_tagid))
        page = request.GET.get("page")
        print("list",page)
        page_object = Pagination3(request, queryset)
        context = {
            "queryset": page_object.page_queryset,  # 取数据
            "page_string": page_object.html(),  # 取页码
            "sel_tagid": sel_tagid,
            "tagid": tagid,
            "vals1": vals1,
            "width": width,
            "page": page

        }
        return render(request, "index_mark.html", context)

def index_mark_edit(request, page,stage,eval,slug,nid):
    """编辑指标"""
    print(stage,eval)
    if request.method == "GET":
        # 展示部分
        context = {
            "nid":nid,
            "slug":slug,
            "stage":stage,
            "eval":eval,
            "page":page
        }
        return render(request, "index_mark_edit.html",context)
    else:
        stage = request.POST['stage'][0]
        eval = request.POST['eval'][0]
        editpos(str(slug),str(nid),str(stage),str(eval))
        tagid = fluxdbOperator().get_plane_measurements()
        vals1 = ["frameId", "stage", "eval"]
        width = 'width:' + str(1 / len(vals1)) + '%'
        client = fluxdbOperator()
        result = client.select_num_battle(str(slug))
        queryset = []
        for i in range(0, len(result), 50):
            queryset.append(result[i])
        page = request.POST.get("page")
        print("edit",page)
        page_object = Pagination3(request, queryset)
        context = {
            "queryset": page_object.page_queryset,  # 取数据
            "page_string": page_object.html(),  # 取页码
            "sel_tagid": slug,
            "tagid": tagid,
            "vals1": vals1,
            "width": width,
            "page":page

        }
        return render(request, "index_mark.html", context)

#外部算法调用
def al_run(request):
    tagid = fluxdbOperator().get_plane_measurements()
    sel_tagid = request.POST.get("tagid")
    # get请求，展示基础页面
    if not (sel_tagid):
        context = {
            "tagid": tagid,
            "x1": [],
            "x2": [],
            "red1": [],
            "red2": [],
            "red3": [],
            "red4": [],
            "blue1": [],
            "blue2": [],
            "blue3": [],
            "blue4": [],
        }
        return render(request, "al_run.html", context)
    else:
        # 文件列表
        files = request.FILES.getlist('file')
        # 删除unzip下的所有文件
        for dir_list in os.listdir(readConfig().getUnZipRootPath()):
            shutil.rmtree(os.path.join(readConfig().getUnZipRootPath(), dir_list))
        del_file(readConfig().getZipRootPath())
        # 得到模型名称
        filename = ''
        for file in files:
            des = open(os.path.join(readConfig().getZipRootPath(), file.name), 'wb+')
            print("shoudao")
            for chunk in file.chunks():
                des.write(chunk)
            des.close()
            filename = os.path.join(readConfig().getZipRootPath(), file.name)
        #使用模型
        measurement = str(sel_tagid)
        # 创建数据集
        from PredictionModel.outcall import OUTCALL
        red1, red2, red3, red4, blue1, blue2, blue3, blue4 = OUTCALL(filename,measurement).outcall()
        x1 = [i + 20 for i in range(len(red1))]
        x2 = [i + 20 for i in range(len(blue1))]
        context = {
            "sel_tagid": sel_tagid,
            "tagid": tagid,
            "x1": x1,
            "x2": x2,
            "red1": red1,
            "red2": red2,
            "red3": red3,
            "red4": red4,
            "blue1": blue1,
            "blue2": blue2,
            "blue3": blue3,
            "blue4": blue4,
        }
        return render(request, "al_run.html", context)
