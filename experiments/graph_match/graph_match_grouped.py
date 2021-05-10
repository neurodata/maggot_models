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
#%%
mg.nodes["inds"] = range(len(mg.nodes))
nodes = mg.nodes.copy()
adj = mg.sum.adj.copy()
left_nodes = nodes[nodes["hemisphere"] == "L"].copy()
left_paired_nodes = left_nodes[left_nodes["pair_id"] > -1].copy()
left_unpaired_nodes = left_nodes[left_nodes["pair_id"] <= -1].copy()
# left_nodes.sort_values("pair_id", inplace=True, ascending=False)
right_nodes = nodes[nodes["hemisphere"] == "R"].copy()
right_paired_nodes = right_nodes[right_nodes["pair_id"] > -1].copy()
right_unpaired_nodes = right_nodes[right_nodes["pair_id"] <= -1].copy()

left_paired_nodes.sort_values("pair_id", inplace=True)
right_paired_nodes.sort_values("pair_id", inplace=True)
assert (right_paired_nodes["pair"] == left_paired_nodes.index).all()


left_paired_inds = left_paired_nodes["inds"]
right_paired_inds = right_paired_nodes["inds"]
n_pairs = len(left_paired_inds)
seeds = np.arange(n_pairs)
# lup_inds = left_unpaired_nodes["inds"]
# rup_inds = right_unpaired_nodes["inds"]
# left_inds = np.concatenate((lp_inds, lup_inds))
# right_inds = np.concatenate((rp_inds, rup_inds))
# ll_adj = adj[np.ix_(left_inds, left_inds)]
# rr_adj = adj[np.ix_(right_inds, right_inds)]
# seeds = np.arange(n_pairs)

# print("Pairs all valid: ")
# print((nodes.iloc[lp_inds].index == nodes.iloc[rp_inds]["pair"]).all())

#%%

left_idx = left_nodes.index
right_idx = right_nodes.index
nodes["predicted_pair"] = -1
nodes["predicted_pair_id"] = -1
nodes.loc[left_paired_inds.index, "predicted_pair"] = right_paired_inds.index
nodes.loc[right_paired_inds.index, "predicted_pair"] = left_paired_inds.index
paired_nodes = nodes[nodes["pair_id"] > 1]
nodes.loc[paired_nodes.index, "predicted_pair_id"] = paired_nodes["pair_id"]
max_pair_id = nodes["predicted_pair_id"].max()

max_iter = 30
n_init = 100
match_classes = [
    "sens-AN",
    "sens-MN",
    "sens-photoRh5",
    "sens-photoRh6",
]

for mc in match_classes[:]:
    left_target_inds = left_nodes[left_nodes["merge_class"] == mc]["inds"]
    print(f"Number of left targets: {len(left_target_inds)}")
    right_target_inds = right_nodes[right_nodes["merge_class"] == mc]["inds"]
    print(f"Number of right targets: {len(right_target_inds)}")
    left_inds = np.concatenate((left_paired_inds, left_target_inds))
    right_inds = np.concatenate((right_paired_inds, right_target_inds))
    ll_adj = adj[np.ix_(left_inds, left_inds)]
    rr_adj = adj[np.ix_(right_inds, right_inds)]
    sizes = (len(ll_adj), len(rr_adj))
    smaller = np.argmin(sizes)
    larger = np.setdiff1d((0, 1), (smaller))[0]
    adjs = (ll_adj, rr_adj)
    smaller_adj = adjs[smaller]
    larger_adj = adjs[larger]
    target_inds = (left_target_inds, right_target_inds)
    smaller_target_inds = target_inds[smaller]
    larger_target_inds = target_inds[larger]
    gm = GraphMatch(
        n_init=n_init,
        init="barycenter",
        eps=1e-2,
        max_iter=max_iter,
        shuffle_input=True,
    )
    gm.fit(smaller_adj, larger_adj, seeds_A=seeds, seeds_B=seeds)
    nonseed_perm_inds = gm.perm_inds_[n_pairs : len(smaller_adj)] - n_pairs
    smaller_target_idx = smaller_target_inds.index
    larger_target_idx = larger_target_inds.index[nonseed_perm_inds]
    nodes.loc[smaller_target_idx, "predicted_pair"] = larger_target_idx
    nodes.loc[larger_target_idx, "predicted_pair"] = smaller_target_idx
    pair_ids = np.arange(max_pair_id + 1, max_pair_id + 1 + len(smaller_target_idx))
    nodes.loc[smaller_target_idx, "predicted_pair_id"] = pair_ids
    nodes.loc[larger_target_idx, "predicted_pair_id"] = pair_ids
    max_pair_id = max(pair_ids)

# check that everything worked
paired_nodes = nodes[nodes["predicted_pair"] > 1].copy()
paired_nodes.sort_values("predicted_pair_id", inplace=True)
left_paired_nodes = paired_nodes[paired_nodes["left"]]
right_paired_nodes = paired_nodes[paired_nodes["right"]]
assert (left_paired_nodes["predicted_pair"] == right_paired_nodes.index).all()
assert (right_paired_nodes["predicted_pair"] == left_paired_nodes.index).all()


join_node_meta(
    nodes["predicted_pair"],
    check_collision=False,
    overwrite=True,
)

join_node_meta(
    nodes["predicted_pair_id"],
    check_collision=False,
    overwrite=True,
)

# new_prediction_nodes = nodes[(nodes["pair"] < 2) & (nodes["predicted_pair"] != -1)][
#     ["merge_class", "hemisphere", "predicted_pair"]
# ].copy()
# new_prediction_nodes = new_prediction_nodes.sort_values(["merge_class", "hemisphere"])
# new_prediction_nodes.to_csv("new_prediction_nodes.csv")


#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
