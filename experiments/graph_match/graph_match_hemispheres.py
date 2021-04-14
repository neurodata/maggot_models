#%%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from graspy import __version__ as graspologic_version
from graspy.match import GraphMatch
from src.data import load_metagraph
from src.graph import MetaGraph
from src.io import savecsv, savefig
from src.visualization import adjplot, set_theme

set_theme()

FNAME = os.path.basename(__file__)[:-3]
print(f"Current file: {FNAME}")

print(f"Graspologic version: {graspologic_version}")


def stashfig(name, **kws):
    savefig(
        name,
        save_on=True,
        fmt="pdf",
        pathname="./maggot_models/experiments/graph_match/figs",
        **kws,
    )


def stashcsv(df, name, **kws):
    savecsv(
        df,
        name,
        pathname="./maggot_models/experiments/graph_match/outs",
        **kws,
    )


# %% [markdown]
# ## Load and preprocess data
graph_type = "G"
master_mg = load_metagraph(graph_type)
mg = MetaGraph(master_mg.adj, master_mg.meta)
meta = mg.meta
meta = meta[meta["paper_clustered_neurons"] | meta["accessory_neurons"]].copy()
mg.reindex(meta.index, use_ids=True)
meta = mg.meta.copy()

#%%
temp_meta = meta[meta["left"] | meta["right"]]
unpair_idx = temp_meta[~temp_meta["pair"].isin(temp_meta.index)].index

subcols = ["merge_class", "hemisphere", "pair", "pair_id", "paired"]
meta["paired"] = True
meta.loc[unpair_idx, "paired"] = False
print("There are some neurons with invalid pairs:")
print(meta.loc[unpair_idx][meta.loc[unpair_idx, "pair"] != -1][subcols])
print("Setting those pairs to -1 for now")
meta.loc[unpair_idx, ["pair", "pair_id"]] = -1

#%%
left_idx = meta[meta["left"]].index
print(f"Nodes on left: {len(left_idx)}")
left_mg = MetaGraph(mg.adj, mg.meta)
left_mg = left_mg.reindex(left_idx, use_ids=True)
left_mg = left_mg.sort_values(["pair_id"], ascending=False)

right_idx = meta[meta["right"]].index
print(f"Nodes on right: {len(right_idx)}")
right_mg = MetaGraph(mg.adj, mg.meta)
right_mg = right_mg.reindex(right_idx, use_ids=True)
right_mg = right_mg.sort_values(["pair_id"], ascending=False)
# NOTE I am chopping off some verts on the right side here, randomly
print("Chopping off some nodes on right...")
right_mg = right_mg.reindex(right_mg.meta.index[: len(left_mg)], use_ids=True)

assert (right_mg.meta["pair_id"].values == left_mg.meta["pair_id"].values).all()

# %% [markdown]
# ## Set up graph matching
n_pairs = len(right_mg.meta[right_mg.meta["pair_id"] != -1])
print(f"Number of known and valid pairs: {n_pairs}")
left_adj = left_mg.adj
right_adj = right_mg.adj
left_seeds = right_seeds = np.arange(n_pairs)
n_init = 50

# %% [markdown]
# ## Run graph matching
np.random.seed(8888)
currtime = time.time()
gms = []
scores = []
for i in tqdm(range(n_init)):
    gm = GraphMatch(n_init=1, init_method="barycenter", eps=0.5, shuffle_input=True)
    gm.fit(left_adj, right_adj, seeds_A=left_seeds, seeds_B=right_seeds)
    gms.append(gm)
    scores.append(gm.score_)
print(f"{(time.time() - currtime)/60:0.2f} minutes elapsed for graph matching")

# %% [markdown]
# ## Plot performance and choose the best
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.distplot(scores, ax=ax, rug=True)
max_ind = np.argmax(scores)
gm = gms[max_ind]
ax.axvline(np.max(scores), linewidth=1.5, color="black", linestyle="--")
ax.set(xlabel="Graph match score")
stashfig("gm-scores")

# %% [markdown]
# ## Plot the predicted matching
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
perm_inds = gm.perm_inds_
true_pairs = np.array(n_pairs * ["paired"] + (len(left_adj) - n_pairs) * ["unpaired"])
pal = sns.color_palette("deep", 3)
color_dict = dict(zip(["unpaired", "paired"], pal[1:]))
_, _, top, _ = adjplot(
    left_adj,
    ax=axs[0],
    plot_type="scattermap",
    sizes=(1, 1),
    colors=true_pairs,
    palette=color_dict,
    color=pal[0],
    ticks=True,
)
top.set_title("Left")
_, _, top, _ = adjplot(
    right_adj[np.ix_(perm_inds, perm_inds)],
    ax=axs[1],
    plot_type="scattermap",
    sizes=(1, 1),
    colors=true_pairs,
    palette=color_dict,
    color=pal[0],
    ticks=True,
)
top.set_title("Right")
plt.tight_layout()
stashfig("gm-results-adj")

#%%
# || A^T P B P^T||_F
A = left_adj
PBPT = right_adj[np.ix_(perm_inds, perm_inds)]
obj_func_by_node = np.diag(A.T @ PBPT)
true_pair_obj_funcs = obj_func_by_node[true_pairs == "paired"]
pred_pair_obj_funcs = obj_func_by_node[true_pairs == "unpaired"]

fig, axs = plt.subplots(2, 1, figsize=(8, 8))
bins = np.linspace(0, 5, 25)
ax = axs[0]
sns.distplot(
    np.log10(true_pair_obj_funcs + 1),
    ax=ax,
    label="Paired",
    kde=False,
    norm_hist=True,
    bins=bins,
)
sns.distplot(
    np.log10(pred_pair_obj_funcs + 1),
    ax=ax,
    label="Unpaired",
    kde=False,
    norm_hist=True,
    bins=bins,
)
ax.set(ylabel="Density")
ax.legend()
ax = axs[1]
sns.distplot(
    np.log10(true_pair_obj_funcs + 1),
    ax=ax,
    label="Paired",
    hist_kws=dict(cumulative=True),
    kde=False,
    norm_hist=True,
    bins=bins,
)
sns.distplot(
    np.log10(pred_pair_obj_funcs + 1),
    ax=ax,
    label="Unpaired",
    hist_kws=dict(cumulative=True),
    kde=False,
    norm_hist=True,
    bins=bins,
)
ax.set(xlabel="log(Per node graph match score)", ylabel="Cumulative density")
stashfig("gm-node-score-eval")

# %% [markdown]
# ## Apply the pairs
left_pair_id = pd.Series(index=left_mg.meta.index, data=np.arange(len(left_mg)))
right_pair_idx = right_mg.meta.index[perm_inds]
right_pair_id = pd.Series(index=right_pair_idx, data=np.arange(len(right_mg)))
pair_id = pd.concat((left_pair_id, right_pair_id))
mg = load_metagraph(graph_type)
mg = mg.reindex(pair_id.index, use_ids=True)
mg.meta["predicted_pair_id"] = pair_id

n_predicted_pairs = len(left_pair_id)
left_ids = mg.meta.iloc[:n_predicted_pairs].index.values
right_ids = mg.meta.iloc[n_predicted_pairs:].index.values
left_pair = pd.Series(index=left_mg.meta.index, data=right_ids)
right_pair = pd.Series(index=right_pair_idx, data=left_ids)
pair = pd.concat((left_pair, right_pair))
mg.meta["predicted_pair"] = pair


# %% [markdown]
# ## Save the output somehow
stashcsv(
    mg.meta,
    "pair_meta",
)
