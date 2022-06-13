import pandas as pd


def community_data_op(choice):
    dk_list = ['诱饵僚机（电磁）-假目标数量', '无人制空僚机（电子战）-最小干扰距离', '无人制空僚机（电子战）-最大干扰角度', '无人制空僚机（电子战）-压制效果', '无人预警僚机-最小干扰距离', '无人预警僚机-最大干扰角度', '无人预警僚机-压制效果']

    gj_list = ['光电探测僚机-最大截获距离', '光电探测僚机-最大截获角度', '无人制空僚机（雷达）-最大截获距离', '无人制空僚机（雷达）-最大截获角度', '无人制空僚机-导弹数量', '无人制空僚机-最大攻击距离', '无人制空僚机-最小攻击距离']

    pd_reader = pd.read_csv("./app0/gongji.csv", encoding='GBK')
    namedic = {}
    if(choice in dk_list):
        pd_reader = pd.read_csv("./app0/duikang.csv", encoding='GBK')
        name0 = pd_reader.get('名称')[:]
        namelist = name0.values.tolist()
        for i in range(len(namelist)):
            namedic[i+1] = namelist[i]
    else:
        pd_reader = pd.read_csv("./app0/gongji.csv", encoding='GBK')
        name0 = pd_reader.get('名称')[:]
        namelist = name0.values.tolist()
        for i in range(len(namelist)):
            namedic[i+1] = namelist[i]
    # names = pd_reader.get('名称')
    value_list = pd_reader.get(choice)
    # min_value = min(value_list, key=abs)
    max_value = max(value_list, key=abs)
    if(abs(max_value)>0.1):
        bias = 1000
    else:
        bias = 5000

    # 前端传入所选tage_name
    # values = pd_reader.get('诱饵僚机（光电）-最大探测距离')

    # res = {}


    nodes = []
    # node = {'id': index}
    links = []
    categories = [
        {
            "name": choice
        },
        {
            "name": "正相关"
        },
        {
            "name": "负相关"
        },
        {
            "name": "无相关"
        }
    ]
    for index in range(len(pd_reader.get('名称')) + 1):
        if index == 0:
            node = {
                "id": index,
                # "name": '诱饵僚机（光电）-最大探测距离',
                'name': choice,
                "symbolSize": 500.12381,
                "x": 0,
                "y": 0,
                "value": 0,
                "category": 0
            }
            nodes.append(node)
            continue

        name = pd_reader.get('名称')[index - 1]
        # value = pd_reader.get('诱饵僚机（光电）-最大探测距离')[index - 1]
        value = pd_reader.get(choice)[index - 1]
        if abs(value) > 0.4:
            bias = 500
        if value == 0:
            node = {
                "id": index,
                "name": index,
                "symbolSize": 50.312381,
                "x": 0,
                "y": 0,
                "value": value,
                "category": 3
            }
            nodes.append(node)
        else:
            node = {
                "id": index,
                "name": index,
                "symbolSize": 100.12381 + abs(value) * bias,
                "x": 0,
                "y": 0,
                "value": value,
                "category": 1 if value > 0 else 2
            }
            nodes.append(node)
            link = {
                "source": str(0),
                "target": str(index),
                "label": {
                    "normal": {
                        "show": 'true',
                        "formatter": str(value),
                    }
                },
            }
            links.append(link)
        # nodes[]

    # res = {}
    # res.s

    res = {'nodes': nodes, 'links': links, 'categories': categories}
    # print(a)
    # print(b)
    # print([a,b])
    return res,namedic
