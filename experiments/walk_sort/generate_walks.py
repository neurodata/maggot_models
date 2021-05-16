#%%
import numpy as np
from src.data import load_maggot_graph
from src.io import save_walks
from src.traverse import RandomWalk, to_markov_matrix
from tqdm import tqdm


def get_nodes(meta, labels):
    nodes = []
    for i, (node, row) in enumerate(meta.iterrows()):
        merge_class = row["simple_group"]
        for label in labels:
            if label in merge_class:
                nodes.append(i)
    return np.unique(nodes)


edge_type = "ad"
outdir = "./maggot_models/experiments/walk_sort/outs/"
n_init = 256
max_hops = 16
allow_loops = False
reverse = False

start_labels = [("sensories",), ("ascendings",)]


stop_labels = [("dVNC",), ("dSEZ",), ("RGN",), ("motor",)]


mg = load_maggot_graph()
mg.to_largest_connected_component()
adj = mg.to_edge_type_graph(edge_type).adj
meta = mg.nodes
meta["inds"] = range(len(meta))

# # create a dataframe of metadata
# node_data = dict(g.nodes(data=True))
# meta = pd.DataFrame().from_dict(node_data, orient="index")
# meta = meta.sort_index()
# meta["inds"] = range(len(meta))
# nodelist = meta.index.values

# # grab the adjacency matrix
# adj = nx.to_numpy_array(g, nodelist)

if reverse:
    adj = adj.T
    start_labels, stop_labels = stop_labels, start_labels

transition_probs = to_markov_matrix(adj)  # row normalize!

n_class_pairs = len(start_labels) * len(stop_labels)
i = 0
all_walks = []
for start_keys in start_labels:
    # start_nodes = meta[meta["merge_class"].isin(start_keys)]["inds"].values
    start_nodes = get_nodes(meta, start_keys)
    for stop_keys in stop_labels:
        # stop_nodes = meta[meta["merge_class"].isin(stop_keys)]["inds"].values
        stop_nodes = get_nodes(meta, stop_keys)
        rw = RandomWalk(
            transition_probs,
            stop_nodes=stop_nodes,
            max_hops=max_hops,
            allow_loops=allow_loops,
        )
        sensorimotor_walks = []  # walks for this s-m pair
        for node in tqdm(
            start_nodes, desc=f"Generating walks for {start_keys} to {stop_keys}"
        ):
            for _ in range(n_init):
                rw.start(node)
                walk = rw.traversal_
                if walk[-1] in stop_nodes:  # only keep ones that made it to output
                    walk = list(meta.index[walk])
                    sensorimotor_walks.append(walk)
        i += 1
        msg = (
            f"({i} / {n_class_pairs}) Completed: "
            + f"{start_keys} to {stop_keys}: {len(sensorimotor_walks)} walks - "
            + f"{len(start_nodes)} to {len(stop_nodes)} neurons"
        )
        print(msg)
        all_walks.append(sensorimotor_walks)

outfile = f"walks-gt={edge_type}-n_init={n_init}-hops={max_hops}-loops={allow_loops}-reverse={reverse}.txt"
save_walks(all_walks, name=outfile, outpath=outdir, multidoc=True)
