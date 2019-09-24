#%%
from src.data import load_june
from src.utils import meta_to_array
from graspy.utils import import_graph
import networkx as nx

graph = load_june("Gadn")
names = meta_to_array(graph, "Name")
names

classes = meta_to_array(graph, "Class")
classes

nodes = list(graph.nodes())

labels = dict(zip(nodes, names))

for i, c in enumerate(classes):
    if c == "OANs":
        print(names[i])
        print(i)
        print

adj = import_graph(graph)
#%%
np.unique(classes)

#%%
for i, n in enumerate(graph.nodes(data=True)):
    data = n[1]
    node = n[0]
    name = data["Name"]
    cell_class = data["Class"]
    if cell_class == "KCs":
        print(name)
        print(i)
        print(node)
        print("Edges incident: " + str(len(graph[node])))
        neighbor_graph = nx.ego_graph(graph, node)
        neighbor_names = meta_to_array(neighbor_graph, "Name")
        neighbor_nodes = list(neighbor_graph.nodes())
        labels = dict(zip(neighbor_nodes, neighbor_names))
        plt.figure(figsize=(10, 10))
        nx.draw_networkx(neighbor_graph, labels=labels)
        plt.title(name)
        plt.show()
oan_ind = "n2243"

#%%
for nbr in graph[oan_ind]:
    print(nbr)
#%%
graph.subgraph(["n1"])
#%%
oan_ind = "n2243"


out = graph.nodes("n2243")
out
for i in out:
    print(i)
#%%
graph["n2243"]

#%%
