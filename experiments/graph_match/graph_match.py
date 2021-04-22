#%%
import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.match import GraphMatch
from src.data import join_node_meta, load_maggot_graph
from src.io import savefig
from src.visualization import adjplot, set_theme
from tqdm import tqdm

t0 = time.time()

set_theme()

FNAME = os.path.basename(__file__)[:-3]
print(f"Current file: {FNAME}")


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/graph_match/figs",
        **kws,
    )


# %% [markdown]
# ## Load and preprocess data

mg = load_maggot_graph()
mg = mg[mg.nodes["paper_clustered_neurons"] | mg.nodes["accessory_neurons"]]
mg = mg[mg.nodes["hemisphere"].isin(["L", "R"])]
mg.to_largest_connected_component(verbose=True)

# from giskard.utils import get_paired_inds

# left_inds, right_inds = get_paired_inds(mg.nodes)

mg.nodes["_inds"] = range(len(mg.nodes))
nodes = mg.nodes
adj = mg.sum.adj
left_nodes = nodes[nodes["hemisphere"] == "L"].copy()
left_paired_nodes = left_nodes[left_nodes["pair_id"] > -1]
left_unpaired_nodes = left_nodes[left_nodes["pair_id"] <= -1]
# left_nodes.sort_values("pair_id", inplace=True, ascending=False)
right_nodes = nodes[nodes["hemisphere"] == "R"].copy()
right_paired_nodes = right_nodes[right_nodes["pair_id"] > -1]
right_unpaired_nodes = right_nodes[right_nodes["pair_id"] <= -1]

assert (right_paired_nodes["pair"].isin(left_paired_nodes.index)).all()
lp_inds = left_paired_nodes.loc[right_paired_nodes["pair"]]["_inds"]
rp_inds = right_paired_nodes["_inds"]
n_pairs = len(rp_inds)
lup_inds = left_unpaired_nodes["_inds"]
rup_inds = right_unpaired_nodes["_inds"]
left_inds = np.concatenate((lp_inds, lup_inds))
right_inds = np.concatenate((rp_inds, rup_inds))
ll_adj = adj[np.ix_(left_inds, left_inds)]
rr_adj = adj[np.ix_(right_inds, right_inds)]
seeds = np.arange(n_pairs)

print("Pairs all valid: ")
print((nodes.iloc[lp_inds].index == nodes.iloc[rp_inds]["pair"]).all())


# %% [markdown]
# ## Run graph matching
print(f"Number of known and valid pairs: {n_pairs}")
n_init_rand = 100
n_init_bary = 100
max_iter = 30
np.random.seed(8888)
currtime = time.time()
gms = []
scores = []
for i in tqdm(range(n_init_rand)):
    gm = GraphMatch(
        n_init=1, init="rand", eps=1e-2, max_iter=max_iter, shuffle_input=True
    )
    gm.fit(ll_adj, rr_adj, seeds_A=seeds, seeds_B=seeds)
    gms.append(gm)
    scores.append(gm.score_)

for i in tqdm(range(n_init_bary)):
    gm = GraphMatch(
        n_init=1, init="barycenter", eps=1e-2, max_iter=max_iter, shuffle_input=True
    )
    gm.fit(ll_adj, rr_adj, seeds_A=seeds, seeds_B=seeds)
    gms.append(gm)
    scores.append(gm.score_)
print(f"{(time.time() - currtime)/60:0.2f} minutes elapsed for graph matching")

# %% [markdown]
# ## Plot performance and choose the best
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.histplot(scores, ax=ax)
max_ind = np.argmax(scores)
gm = gms[max_ind]
ax.axvline(np.max(scores), linewidth=1.5, color="black", linestyle="--")
ax.set(xlabel="Graph match score")
stashfig("gm-scores")

# %% [markdown]
# ## Plot the predicted matching
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
perm_inds = gm.perm_inds_
true_pairs = np.array(n_pairs * ["paired"] + (len(ll_adj) - n_pairs) * ["unpaired"])
pal = sns.color_palette("deep", 3)
color_dict = dict(zip(["unpaired", "paired"], pal[1:]))
_, _, top, _ = adjplot(
    ll_adj,
    ax=axs[0],
    plot_type="scattermap",
    sizes=(1, 1),
    colors=true_pairs,
    palette=color_dict,
    color=pal[0],
    ticks=True,
)
top.set_title("Left")
true_pairs = np.array(n_pairs * ["paired"] + (len(rr_adj) - n_pairs) * ["unpaired"])
_, _, top, _ = adjplot(
    rr_adj[np.ix_(perm_inds, perm_inds)],
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
valid_perm_inds = perm_inds[: len(ll_adj)]
A = ll_adj
PBPT = rr_adj[np.ix_(valid_perm_inds, valid_perm_inds)]
A_node_norms = (np.linalg.norm(A, axis=0) + np.linalg.norm(A, axis=1)) / 2
B_node_norms = (np.linalg.norm(PBPT, axis=0) + np.linalg.norm(PBPT, axis=1)) / 2
node_norms = A_node_norms * B_node_norms
obj_func_by_node = np.diag(A.T @ PBPT) / node_norms
true_pairs = np.array(n_pairs * ["paired"] + (len(ll_adj) - n_pairs) * ["unpaired"])
true_pair_obj_funcs = obj_func_by_node[true_pairs == "paired"]
pred_pair_obj_funcs = obj_func_by_node[true_pairs == "unpaired"]

fig, axs = plt.subplots(2, 1, figsize=(8, 8))
bins = np.linspace(0, 1, 50)
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
left_nodes = nodes.iloc[left_inds]
right_nodes = nodes.iloc[right_inds]
left_idx = left_nodes.index
right_idx = right_nodes.index
nodes["predicted_pair"] = -1
nodes["predicted_pair_id"] = -1
right_idx_perm = right_idx[valid_perm_inds]

pair_ids = np.arange(len(right_idx_perm))
nodes.loc[left_idx, "predicted_pair"] = right_idx_perm.values
nodes.loc[right_idx_perm, "predicted_pair"] = left_idx.values
nodes.loc[left_idx, "predicted_pair_id"] = pair_ids
nodes.loc[right_idx_perm, "predicted_pair_id"] = pair_ids

#%%
join_node_meta(
    nodes[["predicted_pair", "predicted_pair_id"]],
    check_collision=False,
    overwrite=True,
)
predicted_nodes = nodes[nodes["predicted_pair_id"] != -1]
has_predicted_matching = pd.Series(
    data=np.ones(len(predicted_nodes.index), dtype=bool),
    index=predicted_nodes.index,
    name="has_predicted_matching",
)
join_node_meta(has_predicted_matching, overwrite=True, fillna=False)

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
