#%%
import numpy as np
from src.data import join_node_meta, load_maggot_graph

mg = load_maggot_graph()


nodes = mg.nodes
nodes = nodes[nodes["has_embedding"]]

n_components = 10
min_split = 32
cols = [
    f"dc_level_{i}_n_components={n_components}_min_split={min_split}" for i in range(8)
]
data = nodes[cols].copy()
data += 1

new_cols = [
    f"dc_level_{i + 1}_n_components={n_components}_min_split={min_split}"
    for i in range(8)
]

data.columns = new_cols

data[f"dc_level_0_n_components={n_components}_min_split={min_split}"] = np.zeros(
    len(data), dtype=int
)

# join_node_meta(data, overwrite=True)
