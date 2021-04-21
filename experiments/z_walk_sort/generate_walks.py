#%%
import click
import networkx as nx
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.io import save_walks
from src.traverse import RandomWalk, to_markov_matrix
from src.data import DATA_DIR, DATA_VERSION


# stop_labels = [
#     ("dVNC", "dSEZ;dVNC", "dVNC;CN", "dVNC;RG"),
#     (
#         "dSEZ",
#         "dSEZ;CN",
#         "dSEZ;LHN",
#         "dSEZ;dVNC",
#     ),
#     (
#         "RGN",
#         "RGN-IPC",
#         "RGN-CA-LP",
#         "RGN-ITP",
#         "dVNC;RGN",
#     ),
#     ("motor-PaN", "motor-MN", "motor-AN", "motor-VAN"),
# ]


def get_nodes(meta, labels):
    nodes = []
    for i, (node, row) in enumerate(meta.iterrows()):
        merge_class = row["merge_class"]
        for label in labels:
            if label in merge_class:
                nodes.append(i)
    return np.unique(nodes)


@click.command()
@click.option(
    "-g",
    "--graph_type",
    default="Gad",
    help="Type of graph to use for random walks",
)
@click.option(
    "-o",
    "--outdir",
    default="./maggot_models/experiments/walk_sort/outs/",
    help="Output folder",
)
@click.option(
    "-n",
    "--n_init",
    default=256,
    help="Number of random walks from each start node in each start/stop class pair.",
)
@click.option(
    "-m",
    "--max_hops",
    default=16,
    help="Maximum number of steps for a random walk to consider.",
)
@click.option(
    "-l",
    "--allow_loops",
    default=False,
    help="Whether to allow random walks to visit the same node twice.",
)
@click.option(
    "-r",
    "--reverse",
    default=False,
    help="Whether to run random walks in the opposite direction.",
)
def main(
    graph_type,
    outdir,
    n_init,
    max_hops,
    allow_loops,
    reverse,
    start_labels=None,
    stop_labels=None,
):

    start_labels = [
        ("sens-AN",),
        ("sens-photoRh6", "sens-photoRh5"),
        ("A00c",),
        ("sens-vtd",),
        ("sens-MN",),
        ("sens-thermo",),
        ("sens-ORN",),
    ]
    start_names = ["AN", "Photo", "A00c", "VTD", "MN", "Thermo", "Odor"]

    stop_labels = [
        ("dVNC",),
        ("dSEZ",),
        ("RGN",),
        ("motor-PaN", "motor-MN", "motor-AN", "motor-VAN"),
    ]
    stop_names = ["dVNC", "dSEZ", "RGN", "motor"]

    infile = f"./{DATA_DIR}/{DATA_VERSION}/{graph_type}.graphml"
    # read in graph as networkx object
    g = nx.read_graphml(infile)

    # create a dataframe of metadata
    node_data = dict(g.nodes(data=True))
    meta = pd.DataFrame().from_dict(node_data, orient="index")
    meta = meta.sort_index()
    meta["inds"] = range(len(meta))
    nodelist = meta.index.values

    # grab the adjacency matrix
    adj = nx.to_numpy_array(g, nodelist)

    if reverse:
        adj = adj.T
        start_labels, stop_labels = stop_labels, start_labels
        start_names, stop_names = stop_names, start_names

    transition_probs = to_markov_matrix(adj)  # row normalize!

    n_class_pairs = len(start_labels) * len(stop_labels)
    i = 0
    all_walks = []
    for start_keys, start_name in zip(start_labels, start_names):
        # start_nodes = meta[meta["merge_class"].isin(start_keys)]["inds"].values
        start_nodes = get_nodes(meta, start_keys)
        for stop_keys, stop_name in zip(stop_labels, stop_names):
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
                start_nodes, desc=f"Generating walks for {start_name} to {stop_name}"
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
                + f"{start_name} to {stop_name}: {len(sensorimotor_walks)} walks - "
                + f"{len(start_nodes)} to {len(stop_nodes)} neurons"
            )
            print(msg)
            all_walks.append(sensorimotor_walks)

    outfile = f"walks-gt={graph_type}-n_init={n_init}-hops={max_hops}-loops={allow_loops}-reverse={reverse}.txt"
    save_walks(all_walks, name=outfile, outpath=outdir, multidoc=True)


if __name__ == "__main__":
    main()
