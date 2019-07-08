from os.path import basename
from pathlib import Path

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

from graspy.datasets import load_drosophila_left, load_drosophila_right
from graspy.utils import binarize, symmetrize
from src.models import select_sbm
from src.utils import save_obj

ex = Experiment("Fit SBM")

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
    n_block_try_range = list(range(1, 21))  # 21
    n_components_try_range = list(range(1, 21))

    param_grid = {  # noqa: F841
        "n_components": n_components_try_range,
        "n_blocks": n_block_try_range,
    }

    # Parameters for the experiment
    n_init = 100  # 50  # noqa: F841
    n_jobs = -2  # noqa: F841
    directed = True  # noqa: F841
    co_block = True  # noqa: F841


def run_fit(seed, param_grid, directed, n_init, n_jobs, co_block):
    # run left
    graph = load_drosophila_left()
    if not directed:
        graph = symmetrize(graph, method="avg")
    graph = binarize(graph)
    sbm_left_df = select_sbm(
        graph,
        param_grid,
        directed=directed,
        n_jobs=n_jobs,
        n_init=n_init,
        co_block=co_block,
    )
    save_obj(sbm_left_df, file_obs, "cosbm_left_df")

    # run right
    graph = load_drosophila_right()
    if not directed:
        graph = symmetrize(graph, method="avg")
    graph = binarize(graph)
    sbm_right_df = select_sbm(
        graph,
        param_grid,
        directed=directed,
        n_jobs=n_jobs,
        n_init=n_init,
        co_block=co_block,
    )
    save_obj(sbm_right_df, file_obs, "cosbm_right_df")

    return 0


@ex.automain
def main(seed, param_grid, directed, n_init, n_jobs, co_block):
    seed = 8888
    out = run_fit(seed, param_grid, directed, n_init, n_jobs, co_block)
    return out
