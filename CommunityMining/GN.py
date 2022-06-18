# -*- coding: utf-8 -*-
import json
import networkx as nx
import matplotlib.pyplot as plt
import sys

from networkx.readwrite import json_graph

from CommunityMining.excelOperator import excelOperator


# sys.path.append('../')

class GN_w:
    def __init__(self, G):
        self.G_copy = G.copy()
        self.G = G
        self.partition = [[n for n in G.nodes()]]
        self.all_Q = [0.0]
        self.max_Q = 0.0

    # Using max_Q to divide communities
    def run(self):
        #		#Until there is no edge in the graph
        #        while len(self.G.edges()) != 0:
        #			#Find the most betweenness edge
        #            edge = max(nx.edge_betweenness_centrality(self.G).items(),key=lambda item:item[1])[0]
        #            #Remove the most betweenness edge
        #            self.G.remove_edge(edge[0], edge[1])
        #			#List the the connected nodes
        #            components = [list(c) for c in list(nx.connected_components(self.G))]
        #            if len(components) != len(self.partition):
        #				#compute the Q
        #                cur_Q = self.cal_Q(components, self.G_copy)
        #                if cur_Q not in self.all_Q:
        #                    self.all_Q.append(cur_Q)
        #                if cur_Q > self.max_Q:
        #                    self.max_Q = cur_Q
        #                    self.partition = components
        #
        #        print('-----------the Max Q and divided communities-----------')
        #        print('The number of Communites:', len(self.partition))
        #        print("Communites:", self.partition)
        #        print('Max_Q:', self.max_Q)
        #        return self.partition, self.all_Q, self.max_Q

        while len(self.G.edges()) != 0:
            edges = {}
            edges_betweenness_centrality = nx.edge_betweenness_centrality(self.G)

            for e, ebc in edges_betweenness_centrality.items():
                edge_weight = ebc / self.G.get_edge_data(e[0], e[1])['value']
                edges[e] = edge_weight

            edge = max(edges.items(), key=lambda item: item[1])[0]
            self.G.remove_edge(edge[0], edge[1])
            components = [list(c) for c in list(nx.connected_components(self.G))]
            if len(components) != len(self.partition):
                # compute the Q
                cur_Q = self.cal_Q(components, self.G_copy)
                if cur_Q not in self.all_Q:
                    self.all_Q.append(cur_Q)
                if cur_Q > self.max_Q:
                    self.max_Q = cur_Q
                    self.partition = components

        # print('-----------the Max Q and divided communities-----------')
        # print('The number of Communites:', len(self.partition))
        # print("Communites:", self.partition)
        # print('Max_Q:', self.max_Q)
        return self.partition, self.all_Q, self.max_Q

    # the process of divding the network
    # Return a list containing the result of each division, until each node is a community
    def run_to_all(self):
        divide = []
        all_Q = []
        while len(self.G.edges()) != 0:
            edge = max(nx.edge_betweenness_centrality(self.G).items(), key=lambda item: item[1])[0]
            self.G.remove_edge(edge[0], edge[1])
            components = [list(c) for c in list(nx.connected_components(self.G))]
            if components not in divide:
                divide.append(components)
            cur_Q = self.cal_Q(components, self.G_copy)
            if cur_Q not in all_Q:
                all_Q.append(cur_Q)
        return divide, all_Q

    # Drawing the graph of Q and divided communities
    def draw_Q(self):
        plt.plot([x for x in range(1, len(self.G.nodes) + 1)], self.all_Q)
        my_x_ticks = [x for x in range(1, len(self.G.nodes) + 1)]
        plt.xticks(my_x_ticks)
        plt.axvline(len(self.partition), color='black', linestyle="--")
        # plt.axhline(self.all_Q[3],color='red',linestyle="--")
        plt.show()

    def add_group(self):
        num = 0
        nodegroup = {}
        for partition in self.partition:
            for node in partition:
                nodegroup[node] = {'group': num}
            num = num + 1
        nx.set_node_attributes(self.G_copy, nodegroup)

    def add_category(self):
        num = 0
        nodegroup = {}
        for partition in self.partition:
            for node in partition:
                nodegroup[node] = {'category': num, 'name': node}
            num = num + 1
        nx.set_node_attributes(self.G_copy, nodegroup)

    def to_gml(self, filepath):
        nx.write_gml(self.G_copy, filepath)

    def to_json(self, filepath):
        with open(filepath, 'w') as outfile1:
            outfile1.write(json.dumps(json_graph.node_link_data(self.G_copy)))

        # 1.读入原来的json
        with open(filepath, 'r') as f:
            content = json.load(f)
        del content["directed"]
        del content["multigraph"]
        del content["graph"]
        # 2.更新字典dict
        categories = {}
        a = []
        for i in range(len(self.partition)):
            category = {}
            category['name'] = str(i)
            a.append(category)
        categories['categories'] = a
        content.update(categories)

        # 3.写入
        with open(filepath, 'w') as f:
            json.dump(content, f)

    # Computing the Q
    def cal_Q(self, partition, G):
        m = len(G.edges(None, False))
        a = []
        e = []

        for community in partition:
            t = 0.0
            for node in community:
                t += len([x for x in G.neighbors(node)])
            a.append(t / (2 * m))

        for community in partition:
            t = 0.0
            for i in range(len(community)):
                for j in range(len(community)):
                    if (G.has_edge(community[i], community[j])):
                        t += 1.0
            e.append(t / (2 * m))

        q = 0.0
        for ei, ai in zip(e, a):
            q += (ei - ai ** 2)
        return q


def GN(path):
    e = excelOperator()
    e.read_csv(path)  # 把甲方给的csv转化为gml文件
    G = nx.read_gml('new_graph.gml')
    algorithm = GN_w(G)
    algorithm.run()
    algorithm.add_group()
    algorithm.draw_Q()
    # print(algorithm.max_Q)
    algorithm.to_gml('out.gml')  # 输出通过GN算法处理后的gml
    e.draw_csv('out.gml')  # 画图


def GN2(tag):
    e = excelOperator()
    namedic = e.make_csv(tag)
    e.read_csv('CommunityMining/test.csv')  # 把甲方给的csv转化为gml文件
    G = nx.read_gml('CommunityMining/new_graph.gml')
    algorithm = GN_w(G)
    algorithm.run()
    algorithm.add_category()
    # algorithm.draw_Q()
    # print(algorithm.max_Q)
    algorithm.to_json('CommunityMining/out0.json')  # 输出通过GN算法处理后的
    return namedic
# GN('攻击聚类绝对值.csv')
# G=nx.read_gml('new_graph2.gml')
# algorithm = GN_w(G)
# algorithm.run()
# algorithm.add_category()
# algorithm.to_json('out2.json')
# GN2('test.csv')



