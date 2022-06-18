import csv
import encodings
import os
from itertools import count
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from DataOperator.jsonOperator import jsonOperator as jo
from decimal import Decimal
import json
from DataOperator.mysqlOperator import mysqlOperator as mo


class excelOperator:
    def __init__(self):
        pass

    def make_csv(self,tag):
        indexdic = {
            'disWhenAt': '导弹发射距离',
            'distanceBeforAt': '先敌攻击距离',
            'distanceBeforLock': '先敌截获距离',
            'distanceBeforSe': '先敌发现距离',
            'resumeTimeAt': '杀伤网络恢复时间',
            'resumeTimeDtAbi': '目标恢复时间',
            'resumeTimeDtAct': '目标探测信息分发直径恢复时间',
            'resumeTimeDtPropor': '目标探测信息分发比例恢复时间',
            'resumeTimeEi': '目标压制恢复时间',
            'resumeTimeFake': '目标假目标诱骗恢复时间',
            'resumeTimeLock': '目标截获恢复时间',
            'resumeTimeShortestEi': '压制信息分发最短路径恢复时间',
            'resumeTimeShortestFake': '压制信息分发最短路径恢复时间',
            'resumeTimeShortestLock': '目标截获信息分发最短路径恢复时间',
            # 'tarDetSta':'目标探测稳定性',
            # 'tarLockSta':'目标截获稳定性',
            # 'timeBeforAt':'先敌攻击时间',
            # 'timeBeforLock':'先敌截获时间',
            # 'timeBeforSe':'先敌发现时间',

        }

        initdic = {
            # 'MisMaxAngle' : '最大发射角',
            'MisMaxRange': '最大射程',
            'MisNum': '携弹量',
            # 'SupMaxAngle':'最大干扰角度',
            # 'SupMinDis':'最小压制距离',

        }
        indexkey = list(indexdic.keys())
        initkey = list(initdic.keys())
        column_lst = list(indexdic.values()) + list(initdic.values())
        # print(column_lst)
        dic = {} # 数字：名字
        for i in range(len(column_lst)):
            dic[i] = column_lst[i]
        # print(dic)

        result = mo().queryNewTagStaticData(tag) #查询所有场次
        indexlist = result['Index']
        initlist = result['Frame']
        num = len(initlist)

        unstrtf_lst = []
        for i in range(len(indexkey)):
            xlist = []
            for j in range(num):
                xlist += json.loads(indexlist[j][indexkey[i]])
            unstrtf_lst.append(xlist)
        for i in range(len(initkey)):
            xlist = []
            for j in range(num):
                xlist += json.loads(initlist[j][initkey[i]])
            unstrtf_lst.append(xlist)

        # 计算列表两两间的相关系数
        data_dict = {}  # 创建数据字典，为生成Dataframe做准备
        for col, gf_lst in zip(column_lst, unstrtf_lst):
            data_dict[col] = gf_lst

        unstrtf_df = pd.DataFrame(data_dict)
        cor1 = unstrtf_df.corr(method='spearman')  # 计算相关系数，得到一个矩阵
        print(cor1)
        print(unstrtf_df.columns.tolist())

        outputpath = 'CommunityMining/test.csv'
        cor1.to_csv(outputpath, sep=',', index=True, header=True)

        return dic


    def read_csv(self, location):
        G = nx.Graph()
        col_dict = {}
        with open(location, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)[1:-1]
            len_row = 0

            # skip_header = [2,3,6,7,10,11]
            ## header中就是所有属性的合集 共92个属性，编号0~91
            for head in header:
                # G.add_node(len_row, label=head)
                # G.add_node(len_row,label=len_row)
                col_dict[str(len_row)]=head ## 保存名称
                len_row += 1

            # assert len_row ==  92
            cur_index = -1
            # path = os.path.join('/root/xtzz/CommunityMining/',)
            jo().storeStaticProcessedData(col_dict,'CommunityMining/feature_name.json')
            # print("Finish Saving")
            ## 循环所有行
            for row in reader:
                cur_index += 1
                if cur_index>=len_row:
                    break
                # if cur_index in skip_header:
                #     continue
                content = row[1:cur_index + 1]
                for col_index in range(cur_index):
                    value = abs(float(content[col_index]))
                    # value = float(content[col_index])
                    if value >= 0.05:
                        G.add_edge(cur_index, col_index, value=value)
                    # G.add_edge(header[cur_index], header[col_index], weight=value)

        # NSet = nx.bipartite.sets(G)
        # User = nx.project(G, NSet[1])
        nx.write_gml(G, 'CommunityMining/new_graph.gml')


    def draw_csv(self,path):
        dirs = list(os.listdir('.'))
        # if 'new_graph.gml' not in dirs:
        # self.read_csv(location)
        # G1 = nx.read_gml('new_graph.gml')
        G1 = nx.read_gml(path)
        # group=[]
        # for node in G1.nodes:
        #     if node['group'] not in group:
        groups = set(nx.get_node_attributes(G1, 'group').values())
        mapping = dict(zip(sorted(groups), count()))
        nodes = G1.nodes()
        colors = [mapping[nodes[n]['group']] for n in nodes]
        # labels = {node:int(node) for node in nodes}
        pos = nx.spring_layout(G1)
        nc=nx.draw_networkx(G1,pos=pos,with_labels=True,node_size=100,
                            font_size=8,node_color = colors,
                            edge_color = 'gray',cmap=plt.cm.tab20_r)
        # ec = nx.draw_networkx_edges(G1, pos, alpha=0.2,edge_color='black')
        # nc = nx.draw_networkx_nodes(G1, pos, nodelist=nodes, node_color=colors, label=nodes,
        #                             node_size=100, cmap=plt.cm.jet)

        # nx.draw_networkx_edges(G1,pos=pos1,edge_color='b')
        # plt.colorbar(nc)
        plt.axis('off')
        # plt.savefig('../image/community.png')
        plt.savefig('image/community.png')
        # plt.show()
        # print(G1.nodes)

    def get_feature_name(self,id=-1):
        # if type(id)=='int':
        #     id = str(id)
        col_dict = jo().convertJsonToDict('CommunityMining/feature_name.json')
        # print(col_dict)
        if id>=0:
            return col_dict[str(id)]
        else:
            return list(col_dict.values())


#
# e= excelOperator()
# e.draw_csv('out2.gml')

# print(e.get_feature_name(5))
