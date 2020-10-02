#%% [markdown]
## gzipping a connectome
# > A (perhaps silly) experiment running gzip on a connectome edgelist
#
# - toc: false
# - badges: false
# - categories: [pedigo, graspologic]
# - hide: false
# - search_exclude: false
#%%
import gzip

# edgelist_loc = "maggot_models/data/processed/2020-09-23/G.edgelist"
edgelist_loc = "./renamed_G.edgelist"
with open(edgelist_loc, "rb") as f:
    raw_edgelist = f.read()

raw_kb = len(raw_edgelist) / 1000

gzip_edgelist = gzip.compress(raw_edgelist)
gzip_kb = len(gzip_edgelist) / 1000

print(f"Raw edgelist size: {raw_kb:.2f} kb")
print(f"Gzipped edgelist size: {gzip_kb:.2f} kb")
print(f"Compression ratio: {gzip_kb/raw_kb:.2f}")

# %% [markdown]
import gzip
import networkx as nx

edgelist_loc = "maggot_models/data/processed/2020-09-23/G.edgelist"

g = nx.read_weighted_edgelist(
    edgelist_loc, delimiter=" ", nodetype=int, create_using=nx.DiGraph
)
#%%
import numpy as np

nodelist = np.array(list(sorted(g.nodes())))
degree_map = dict(g.degree(nodelist))
degrees = list(map(degree_map.get, nodelist))
sort_inds = np.argsort(degrees)[::-1]
node_name_remap = dict(zip(nodelist[sort_inds], range(len(g))))
renamed_g = nx.relabel_nodes(g, node_name_remap, copy=True)
nx.write_weighted_edgelist(renamed_g, "./renamed_G.edgelist")


# %% [markdown]
# ##

from neuropull import load_witvilet_2020

elegans_graphs = load_witvilet_2020()

adult_celegans_g1 = elegans_graphs[-2]

# %% [markdown]
# ##
import networkx as nx

g = adult_celegans_g1.copy()
chemical_g = nx.DiGraph(
    ((u, v, e) for u, v, e in g.edges(data=True) if e["type"] == "chemical")
)


def relabel_nodes(g, method="degree"):
    nodelist = np.array(list(sorted(g.nodes())))
    if method == "degree":
        degree_map = dict(g.degree(nodelist))
        degrees = list(map(degree_map.get, nodelist))
        sort_inds = np.argsort(degrees)[::-1]
    elif method == "random":
        sort_inds = np.random.permutation(len(g))
    node_name_remap = dict(zip(nodelist[sort_inds], range(len(g))))
    renamed_g = nx.relabel_nodes(g, node_name_remap, copy=True)
    return renamed_g


def calc_edgelist_n_bytes(g, weight=False):
    if weight:
        data = ["weight"]
    else:
        data = False
    edgelist_lines = ""
    for line in nx.generate_edgelist(g, delimiter=" ", data=data):
        edgelist_lines += line + "\n"
    print(edgelist_lines[:50])
    edgelist_bytes = bytes(edgelist_lines, encoding="utf8")
    n_bytes = len(edgelist_bytes)
    edgelist_bytes_compressed = gzip.compress(edgelist_bytes)
    n_bytes_compressed = len(edgelist_bytes_compressed)
    return n_bytes, n_bytes_compressed


import pandas as pd

rows = []

for weight in [True, False]:
    degree_relabeled_g = relabel_nodes(g, method="degree")
    n_bytes, n_bytes_compressed = calc_edgelist_n_bytes(
        degree_relabeled_g, weight=weight
    )
    rows.append(
        {
            "n_bytes": n_bytes,
            "n_bytes_compressed": n_bytes_compressed,
            "method": "degree",
            "weight": weight,
        }
    )
    for i in range(10):
        random_relabeled_g = relabel_nodes(g, method="random")
        n_bytes, n_bytes_compressed = calc_edgelist_n_bytes(
            random_relabeled_g, weight=weight
        )
        rows.append(
            {
                "n_bytes": n_bytes,
                "n_bytes_compressed": n_bytes_compressed,
                "method": "random",
                "weight": weight,
            }
        )

n_bytes_df = pd.DataFrame(rows)
n_bytes_df["compression_ratio"] = (
    n_bytes_df["n_bytes_compressed"] / n_bytes_df["n_bytes"]
)
n_bytes_df
#%%
import seaborn as sns


sns.stripplot(x="method", y="compression_ratio", data=n_bytes_df)
