#%% Imports and file loading
from pathlib import Path

import pandas as pd
import networkx as nx
import numpy as np
from graspy.plot import gridplot
from src.data import load_networkx

data_path = Path(
    "./maggot_models/data/raw/Maggot-Brain-Connectome/4-color-matrices_Brain"
)

data_date = "2019-09-18-v2"

graph_types = ["axon-axon", "axon-dendrite", "dendrite-axon", "dendrite-dendrite"]

meta_data_file = "brain_meta-data"

input_counts_file = "input_counts"

output_path = Path("maggot_models/data/processed/2019-09-18-v2")

meta_data_path = data_path / data_date / (meta_data_file + ".csv")
meta_data_df = pd.read_csv(meta_data_path, index_col=0)
meta_data_dict = meta_data_df.to_dict(orient="index")
print(meta_data_df.head())

input_counts_path = data_path / data_date / (input_counts_file + ".csv")
input_counts_df = pd.read_csv(input_counts_path, index_col=0)
cols = input_counts_df.columns.values
cols = [str(c).strip(" ") for c in cols]
input_counts_df.columns = cols
print(input_counts_df.head())


def df_to_nx(df, meta_data_dict):
    c = df.columns.values
    c = c.astype(int)
    r = df.index.values
    df.columns = c
    if not (c == r).all():
        raise ValueError("Mismatching df indexing")
    graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
    nx.set_node_attributes(graph, meta_data_dict)
    return graph


#%% Import the raw graphs
nx_graphs_raw = {}
df_graphs_raw = {}
for graph_type in graph_types:
    print(graph_type)
    edgelist_path = data_path / data_date / (graph_type + ".csv")
    adj = pd.read_csv(edgelist_path, index_col=0)
    graph = df_to_nx(adj, meta_data_dict)
    nx_graphs_raw[graph_type] = graph
    df_graphs_raw[graph_type] = adj

    gridplot([adj.values], title=graph_type)
    print()


#%% Normalize weights for the raw graphs
df_graphs_norm = {}
nx_graphs_norm = {}

input_counts = input_counts_df["axon_inputs"].values
input_counts[input_counts == 0] = 1
for graph_type in ["axon-axon", "dendrite-axon"]:
    print(graph_type)
    df_adj_raw = df_graphs_raw[graph_type]
    if (input_counts_df.index.values == adj.index.values).all():
        print("Same indexing!")
    adj_raw = df_adj_raw.values
    adj_norm = adj_raw / input_counts[np.newaxis, :]
    print(adj_norm.sum(axis=0).max())
    df_adj_norm = pd.DataFrame(
        index=df_adj_raw.index, columns=df_adj_raw.columns, data=adj_norm
    )
    df_graphs_norm[graph_type] = df_adj_norm
    graph = df_to_nx(df_adj_norm, meta_data_dict)
    gridplot([df_adj_norm.values], title=graph_type)
    nx_graphs_norm[graph_type] = graph
    print()

input_counts = input_counts_df["dendrite_inputs"].values
input_counts[input_counts == 0] = 1
for graph_type in ["axon-dendrite", "dendrite-dendrite"]:
    print(graph_type)
    df_adj_raw = df_graphs_raw[graph_type]
    if (input_counts_df.index.values == adj.index.values).all():
        print("Same indexing!")
    adj_raw = df_adj_raw.values
    adj_norm = adj_raw / input_counts[np.newaxis, :]
    print(adj_norm.sum(axis=0).max())
    df_adj_norm = pd.DataFrame(
        index=df_adj_raw.index, columns=df_adj_raw.columns, data=adj_norm
    )
    df_graphs_norm[graph_type] = df_adj_norm
    graph = df_to_nx(df_adj_norm, meta_data_dict)
    gridplot([df_adj_norm.values], title=graph_type)
    nx_graphs_norm[graph_type] = graph
    print()


#%% All-all graph
total_input = (
    input_counts_df["dendrite_inputs"].values + input_counts_df["axon_inputs"].values
)
total_input[total_input == 0] = 1

all_adj_raw = np.zeros_like(adj_norm)
for graph_type in graph_types:
    all_adj_raw += df_graphs_raw[graph_type].values

df_all_raw = pd.DataFrame(
    index=df_adj_raw.index, columns=df_adj_raw.columns, data=all_adj_raw
)

nx_all_raw = df_to_nx(df_all_raw, meta_data_dict)

all_adj_norm = all_adj_raw / total_input[np.newaxis, :]
df_all_norm = pd.DataFrame(
    index=df_adj_raw.index, columns=df_adj_raw.columns, data=all_adj_norm
)

nx_all_norm = df_to_nx(df_all_norm, meta_data_dict)

#%% Save

out_graphs = []
[out_graphs.append(i) for i in nx_graphs_raw.values()]
[print(i) for i in nx_graphs_raw.keys()]
save_names = ["Gaa", "Gad", "Gda", "Gdd"]
[out_graphs.append(i) for i in nx_graphs_norm.values()]
[print(i) for i in nx_graphs_norm.keys()]
save_names += ["Gaan", "Gdan", "Gadn", "Gddn"]
out_graphs.append(nx_all_raw)
save_names.append("G")
out_graphs.append(nx_all_norm)
save_names.append("Gn")

for name, graph in zip(save_names, out_graphs):
    nx.write_graphml(graph, output_path / (name + ".graphml"))

#%% verify things are right
for name, graph_wrote in zip(save_names, out_graphs):
    print(name)
    graph_read = nx.read_graphml(output_path / (name + ".graphml"))
    adj_read = nx.to_numpy_array(graph_read)
    adj_wrote = nx.to_numpy_array(graph_wrote)
    print(np.array_equal(adj_read, adj_wrote))
    graph_loader = load_networkx(name)
    adj_loader = nx.to_numpy_array(graph_loader)
    print(np.array_equal(adj_wrote, adj_loader))
    print()


#%%
