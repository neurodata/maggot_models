#%%
from pathlib import Path
from operator import itemgetter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from graspy.plot import heatmap
import pandas as pd
from graspy.utils import binarize

data_path = Path("maggot_models/data/raw/20190615_mw")
base_file_name = "mw_20190615_"
graph_types = ["Gaa", "Gad", "Gdd", "Gda"]
plt.style.use("seaborn-white")

graph_dict = {}
graph_list = []
for graph_type in graph_types:
    file_path = data_path / (base_file_name + graph_type + ".graphml")
    graph = nx.read_graphml(file_path)
    graph_dict[graph_type] = graph
    graph_list.append(graph)
    node_data = list(graph.nodes.data())
    names, data = zip(*node_data)
    meta_df = pd.DataFrame(data)
    classes = meta_df["Class"]
    graph = binarize(graph)

    # heatmap(
    #     graph,
    #     inner_hier_labels=classes,
    #     figsize=(20, 20),
    #     transform="simple-nonzero",
    #     hier_label_fontsize=10,
    # )
    # plt.savefig(graph_type, facecolor="w")

#%%
g = graph_list[0]
right_nodes = [x for x, y in g.nodes(data=True)if y["Hemisphere"] == "right"]
left_subgraph = g.subgraph(right_nodes)

def get_subgraph(graph, feature, key):
    sub_nodes = [node for node, meta in graph.nodes(data=True) if meta[feature] == key]
    return graph.subgraph(sub_nodes)

right_subgraph = get_subgraph(g, "Hemisphere", "left")
right_subgraph.nodes

#%%

for graph in graph_list: 
    right_graph = 
#%%
%config InlineBackend.figure_format = 'png'
from graspy.embed import OmnibusEmbed
from graspy.plot import pairplot
from graspy.plot import heatmap
from graspy.utils import pass_to_ranks

right_graph_list = [get_subgraph(g, "Hemisphere", "left") for g in graph_list]

graphs = right_graph_list
n_graphs = 4
n_verts = len(graphs[0].nodes)
n_components = 4

embed_graphs = [pass_to_ranks(g) for g in graphs]

omni = OmnibusEmbed(n_components=n_components)
latent = omni.fit_transform(embed_graphs)
latent = np.concatenate(latent, axis=-1)
plot_latent = latent.reshape((n_graphs * n_verts, 2 * n_components))
labels = (
    n_verts * ["A -> A"]
    + n_verts * ["A -> D"]
    + n_verts * ["D -> D"]
    + n_verts * ["D -> A"]
)
# latent = np.concatenate(list(latent))
pairplot(plot_latent, labels=labels)
#%%
g = graphs[0]
classes = [meta['Class'] for node, meta in g.nodes(data=True)]
classes = np.array(classes)
unknown = classes == "Other"
plot_unknown = np.tile(unknown, n_graphs)
pairplot(plot_latent, labels=plot_unknown, alpha=0.3)




#%%
