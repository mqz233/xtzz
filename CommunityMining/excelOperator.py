import csv
import encodings
import os
from itertools import count

import networkx as nx
from matplotlib import pyplot as plt
from DataOperator.jsonOperator import jsonOperator as jo
from decimal import Decimal


class excelOperator:
    def __init__(self):
        pass

    def read_csv(self, location):
        G = nx.Graph()
        col_dict = {}
        with open(location, encoding='gbk') as f:
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
            jo().storeStaticProcessedData(col_dict,'../CommunityMining/feature_name.json')
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
                    if value>0:
                        G.add_edge(cur_index, col_index, value=value)
                    # G.add_edge(header[cur_index], header[col_index], weight=value)

        # NSet = nx.bipartite.sets(G)
        # User = nx.project(G, NSet[1])
        nx.write_gml(G, 'new_graph.gml')


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
        plt.savefig('../image/gj_community.png')
        # plt.show()
        # print(G1.nodes)

    def get_feature_name(self,id=-1):
        # if type(id)=='int':
        #     id = str(id)
        col_dict = jo().convertJsonToDict('feature_name.json')
        # print(col_dict)
        if id>=0:
            return col_dict[str(id)]
        else:
            return list(col_dict.values())


#
# e= excelOperator()
# e.draw_csv('out2.gml')
# e.get_feature_name(5)
