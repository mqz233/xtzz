import community
import networkx as nx
import matplotlib.pyplot as plt


# pip install community
# pip install python-louvain
# pip install networkx

#better with karate_graph() as defined in networkx example.
#erdos renyi don't have true community structure
G = nx.read_gml('new_graph.gml')

#first compute the best partition
partition = community.best_partition(G)

#drawing
size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 100,
                                node_color = str(count / size))
nx.draw_networkx_labels(G,pos,font_color='k',font_family='SimHei',alpha=0.8)

nx.draw_networkx_edges(G,pos, alpha=0.5,edge_color='b')
plt.show()
