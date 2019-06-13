from os.path import basename
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver
from graspy.datasets import load_drosophila_left
from graspy.utils import binarize, symmetrize, is_fully_connected
from graspy.plot import heatmap
from src.utils import select_sbm, select_rdpg
import matplotlib.pyplot as plt

ex = Experiment("Drosophila model selection 1")

current_file = basename(__file__)[:-3]

sacred_file_path = Path(f"./maggot_models/models/runs/{current_file}")

slack_obs = SlackObserver.from_config("slack.json")

file_obs = FileStorageObserver.create(sacred_file_path)

ex.observers.append(slack_obs)
ex.observers.append(file_obs)


@ex.config
def config():
    """Variables defined in config get automatically passed to main"""
    n_block_try_range = list(range(1, 11))  # noqa: F841
    n_components_try_range = list(range(1, 13))  # noqa: F841
    n_components_try_rdpg = list(range(1, 13))  # noqa: F841
    directed = True  # noqa: F841
    n_sims_sbm = 50  # noqa: F841


def run_fit(
    seed,
    n_components_try_range,
    n_components_try_rdpg,
    n_block_try_range,
    directed,
    n_sims_sbm,
):
    graph = load_drosophila_left()
    if not directed:
        graph = symmetrize(graph, method="avg")
    graph = binarize(graph)

    connected = is_fully_connected(graph)

    if not connected:
        heatmap(graph)
        plt.show()
        raise ValueError("input graph not connected")

    np.random.seed(seed)

    columns = columns = [
        "n_params_gmm",
        "n_params_sbm",
        "rss",
        "mse",
        "score",
        "n_components_try",
        "n_block_try",
        "sim_ind",
    ]
    sbm_master_df = pd.DataFrame(columns=columns)
    for i in range(n_sims_sbm):
        sbm_df = select_sbm(
            graph, n_components_try_range, n_block_try_range, directed=directed
        )
        sbm_df["sim_ind"] = i
        sbm_master_df = sbm_master_df.append(sbm_df, ignore_index=True, sort=True)
    rdpg_df = select_rdpg(graph, n_components_try_rdpg, directed)
    return (sbm_master_df, rdpg_df)


@ex.automain
def main(
    n_components_try_range,
    n_components_try_rdpg,
    n_block_try_range,
    directed,
    n_sims_sbm,
):
    seed = 8888
    sbm_df, rdpg_df = run_fit(
        seed,
        n_components_try_range,
        n_components_try_rdpg,
        n_block_try_range,
        directed,
        n_sims_sbm,
    )
    return (sbm_df, rdpg_df)
