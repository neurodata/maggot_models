#%%
import datetime
import time

import pandas as pd

from giskard.flow import rank_graph_match_flow, rank_signal_flow, signal_flow
from src.data import join_node_meta, load_maggot_graph

t0 = time.time()


#%%
print("Loading data...")
mg = load_maggot_graph()
nodes = mg.nodes.copy()
mg = mg.sum
mg.to_largest_connected_component(verbose=True)
index = mg.nodes.index
adj = mg.adj
meta = mg.nodes
# #%%
# sort_meta = meta.copy()
# sort_meta.sort_values(
#     ["median_class_visits", "merge_class", "median_node_visits"], inplace=True
# )

# sort_meta["ind"] = range(len(sort_meta))
# color_dict = CLASS_COLOR_DICT
# classes = sort_meta["merge_class"].values
# uni_classes = np.unique(sort_meta["merge_class"])
# class_map = dict(zip(uni_classes, range(len(uni_classes))))
# color_sorted = np.vectorize(color_dict.get)(uni_classes)
# lc = ListedColormap(color_sorted)
# class_indicator = np.vectorize(class_map.get)(classes)
# class_indicator = class_indicator.reshape(len(classes), 1)

# fig, ax = plt.subplots(1, 1, figsize=(1, 10))
# sns.heatmap(
#     class_indicator,
#     cmap=lc,
#     cbar=False,
#     yticklabels=False,
#     # xticklabels=False,
#     square=False,
#     ax=ax,
# )
# ax.axis("off")
# stashfig(f"class-rw-order-heatmap-{walk_spec}")

#%% run signal flow
print("Running signal flow...")
sf = signal_flow(adj)
sf = pd.Series(index=index, data=sf, name="sum_signal_flow")
join_node_meta(sf, overwrite=True)

#%% run rank signal flow
print("Running ranked signal flow...")
rank_sf = rank_signal_flow(adj)
rank_sf = pd.Series(index=index, data=rank_sf, name="sum_rank_signal_flow")
join_node_meta(rank_sf, overwrite=True)

#%% run graph match flow
# print("Running ranked graph match flow...")
# # TODO increase number of inits here
# # TODO use GOAT
# # TODO this is slow AF, should do the numba or scipy version
# rank_gm_flow = rank_graph_match_flow(adj, n_init=1, max_iter=20, eps=1e-2)
# rank_gm_flow = pd.Series(index=index, data=rank_gm_flow, name="sum_rank_gm_flow")
# join_node_meta(rank_gm_flow)

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
