#%%
import pandas as pd

from giskard.flow import rank_graph_match_flow, rank_signal_flow, signal_flow
from src.data import DATA_PATH, DATA_VERSION, load_maggot_graph

#%%
print("Loading data...")
mg = load_maggot_graph()
nodes = mg.nodes.copy()
mg = mg.sum
mg.to_largest_connected_component(verbose=True)
index = mg.nodes.index
adj = mg.adj
#%% run signal flow
print("Running signal flow...")
sf = signal_flow(adj)
sf = pd.Series(index=index, data=sf)
nodes["sum_signal_flow"] = sf

#%% run rank signal flow
print("Running ranked signal flow...")
rank_sf = rank_signal_flow(adj)
rank_sf = pd.Series(index=index, data=rank_sf)
nodes["sum_rank_gm_flow"] = rank_sf

#%% run graph match flow
print("Running ranked graph match flow...")
# TODO increase number of inits here
# TODO use GOAT
# TODO this is slow AF, should do the numba or scipy version
rank_gm_flow = rank_graph_match_flow(adj, n_init=1, max_iter=20, eps=1e-2)
rank_gm_flow = pd.Series(index=index, data=rank_gm_flow)
nodes["sum_rank_gm_flow"] = rank_gm_flow

#%% save
print("Saving to meta_data.csv...")
out_path = DATA_PATH / DATA_VERSION
nodes.sort_index(inplace=True)
nodes.to_csv(out_path / "meta_data.csv")

print("Done!")