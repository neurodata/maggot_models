#%%
import click
import networkx as nx
import pandas as pd
from tqdm import tqdm

from src.io import save_vocab, save_walks
from src.traverse import RandomWalk, to_markov_matrix
from src.data import DATA_DIR, DATA_VERSION

start_labels = [
    ("sens-AN",),
    ("sens-photoRh6", "sens-photoRh5"),
    ("A00c",),
    ("sens-vtd",),
    ("sens-MN",),
    ("sens-PaN",),
    ("sens-thermo",),
    ("sens-ORN",),
]
start_names = ["AN", "Photo", "A00c", "VTD", "MN", "PaN", "Thermo", "Odor"]

stop_labels = [
    ("dVNC", "dSEZ;dVNC", "dVNC;CN", "dVNC;RG"),
    (
        "dSEZ",
        "dSEZ;CN",
        "dSEZ;LHN",
        "dSEZ;dVNC",
    ),
    (
        "RG",
        "RG-IPC",
        "RG-CA-LP",
        "RG-ITP",
        "dVNC;RG",
    ),
    ("motor-PaN", "motor-MN", "motor-AN", "motor-VAN"),
]
stop_names = ["dVNC", "dSEZ", "RG", "motor"]


@click.command()
@click.option(
    "-i",
    "--infile",
    default=f"./{DATA_DIR}/{DATA_VERSION}/G_lcc.graphml",
    help="Location of the maggot graphml file. Must have 'merge_class' as a node field",
)
@click.option(
    "-o",
    "--outdir",
    default="./experiments/maggot_sensorimotor/input/",
    help="Output folder",
)
@click.option(
    "-f",
    "--outfile",
    default="walks.txt",
    help="Output file name",
)
@click.option(
    "-n",
    "--n_init",
    default=64,
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
    default=True,
    help="Whether to allow random walks to visit the same node twice.",
)
def main(
    infile,
    outdir,
    outfile,
    n_init,
    max_hops,
    allow_loops,
):
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
    transition_probs = to_markov_matrix(adj)  # row normalize!

    all_walks = []
    for start_keys, start_name in zip(start_labels, start_names):
        start_nodes = meta[meta["merge_class"].isin(start_keys)]["inds"].values
        for stop_keys, stop_name in zip(stop_labels, stop_names):
            stop_nodes = meta[meta["merge_class"].isin(stop_keys)]["inds"].values
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
                for i in range(n_init):
                    rw.start(node)
                    walk = rw.traversal_
                    if walk[-1] in stop_nodes:  # only keep ones that made it to motor
                        walk = list(meta.index[walk])
                        sensorimotor_walks.append(walk)
            print(
                f"{start_name} to {stop_name}: {len(sensorimotor_walks)} walks - {len(start_nodes)} to {len(stop_nodes)} neurons"
            )
            all_walks.append(sensorimotor_walks)

    save_walks(all_walks, name=outfile, outpath=outdir, multidoc=True)
    save_vocab(g, name=outfile[:-4] + "-vocab.txt", outpath=outdir)


if __name__ == "__main__":
    main()
