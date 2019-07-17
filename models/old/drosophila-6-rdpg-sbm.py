from os.path import basename
from pathlib import Path

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver
from sklearn.model_selection import ParameterGrid

from graspy.datasets import load_drosophila_left
from graspy.utils import binarize, symmetrize
from src.models import select_rdpg, select_sbm, select_dcsbm
from src.utils import save_obj

ex = Experiment("Fit DC")

current_file = basename(__file__)[:-3]

sacred_file_path = Path(f"./maggot_models/models/runs/{current_file}")

slack_obs = SlackObserver.from_config("slack.json")

file_obs = FileStorageObserver.create(sacred_file_path)

ex.observers.append(slack_obs)
ex.observers.append(file_obs)


@ex.config
def config():
    # Variables defined in config get automatically passed to main

    # Parameter range for the models
    n_block_try_range = list(range(1, 11))
    n_components_try_range = list(range(1, 13))
    reg_try_range = np.linspace(0, 10, 10)
    embed_kws_try_range = [{"regularizer": i} for i in reg_try_range]

    param_grid = {  # noqa: F841
        "n_components": n_components_try_range,
        "n_blocks": n_block_try_range,
        "embed_kws": embed_kws_try_range,
    }

    # Parameters for the experiment
    n_init = 50  # 50  # noqa: F841
    n_jobs = -2  # noqa: F841
    directed = True  # noqa: F841


def run_fit(seed, param_grid, directed, n_init, n_jobs):
    graph = load_drosophila_left()
    if not directed:
        graph = symmetrize(graph, method="avg")
    graph = binarize(graph)

    np.random.seed(seed)

    dcsbm_out_df = select_dcsbm(
        graph,
        param_grid,
        directed=directed,
        degree_directed=False,
        n_jobs=n_jobs,
        n_init=n_init,
    )

    ddcsbm_out_df = select_dcsbm(
        graph,
        param_grid,
        directed=directed,
        degree_directed=True,
        n_jobs=n_jobs,
        n_init=n_init,
    )

    save_obj(dcsbm_out_df, file_obs, "dcsbm_out_df")
    save_obj(ddcsbm_out_df, file_obs, "ddcsbm_out_df")
    return 0


@ex.automain
def main(seed, param_grid, directed, n_init, n_jobs):
    seed = 8888
    out = run_fit(seed, param_grid, directed, n_init, n_jobs)
    return out
