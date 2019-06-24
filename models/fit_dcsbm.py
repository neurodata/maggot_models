from os.path import basename
from pathlib import Path

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

from graspy.datasets import load_drosophila_left, load_drosophila_right
from graspy.utils import binarize, symmetrize
from src.models import select_dcsbm
from src.utils import save_obj

ex = Experiment("Fit DSCSBM")

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
    n_block_try_range = list(range(1, 21))
    n_components_try_range = list(range(1, 21))
    reg_try_range = np.linspace(0, 20, 20)
    embed_kws_try_range = [{"regularizer": i} for i in reg_try_range]

    param_grid = {  # noqa: F841
        "n_components": n_components_try_range,
        "n_blocks": n_block_try_range,
        "embed_kws": embed_kws_try_range,
    }

    # Parameters for the experiment
    n_init = 100  # 50  # noqa: F841
    n_jobs = -2  # noqa: F841
    directed = True  # noqa: F841


def run_fit(seed, param_grid, directed, n_init, n_jobs):
    # run left
    graph = load_drosophila_left()
    if not directed:
        graph = symmetrize(graph, method="avg")
    graph = binarize(graph)
    ddcsbm_left_df = select_dcsbm(
        graph,
        param_grid,
        directed=directed,
        degree_directed=False,
        n_jobs=n_jobs,
        n_init=n_init,
    )
    save_obj(ddcsbm_left_df, file_obs, "ddcsbm_left_df")

    # run right
    graph = load_drosophila_right()
    if not directed:
        graph = symmetrize(graph, method="avg")
    graph = binarize(graph)
    ddcsbm_right_df = select_dcsbm(
        graph,
        param_grid,
        directed=directed,
        degree_directed=False,
        n_jobs=n_jobs,
        n_init=n_init,
    )
    save_obj(ddcsbm_right_df, file_obs, "ddcsbm_right_df")

    return 0


@ex.automain
def main(seed, param_grid, directed, n_init, n_jobs):
    seed = 8888
    out = run_fit(seed, param_grid, directed, n_init, n_jobs)
    return out
