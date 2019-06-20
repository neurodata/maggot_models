from os.path import basename
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

from src.utils import gen_B, gen_sbm, save_obj
from src.models import select_sbm

ex = Experiment("SBM model selection")

current_file = basename(__file__)[:-3]

pickle_path = Path("./maggot_models/simulations/outs/")
sacred_file_path = Path(f"./maggot_models/simulations/runs/{current_file}")

slack_obs = SlackObserver.from_config("slack.json")

file_obs = FileStorageObserver.create(sacred_file_path)

ex.observers.append(slack_obs)
ex.observers.append(file_obs)


@ex.config
def config():
    """Variables defined in config get automatically passed to main"""

    n_sims = 2  # noqa: F841
    n_jobs = -2  # noqa: F841
    n_blocks_range = list(range(1, 9))
    n_verts_range = [100]  # [100, 200, 300, 500, 800, 1000]  # noqa: F841
    n_block_try_range = list(range(1, 11))  # noqa: F841
    n_components_try_range = list(range(1, 13))  # noqa: F841

    # keep these the same
    a = 0.1
    b = 0.1

    # how strong to make the block diagonal
    assortivity = 6

    B_mat = gen_B(n_blocks_range[-1], a=a, b=b, assortivity=assortivity)  # noqa: F841

    directed = False  # noqa: F841


def run_sim(
    seed,
    n_blocks_range,
    n_verts_range,
    n_components_try_range,
    n_block_try_range,
    B_mat,
    directed,
):
    np.random.seed(seed)
    columns = [
        "n_params_gmm",
        "n_params_sbm",
        "rss",
        "mse",
        "score",
        "n_components_try",
        "n_block_try",
        "n_blocks",
        "n_verts",
    ]
    master_sbm_df = pd.DataFrame(columns=columns)

    for i, n_blocks in enumerate(n_blocks_range):
        B_mat_trunc = B_mat[:n_blocks, :n_blocks]
        for j, n_verts in enumerate((n_verts_range)):
            graph, labels = gen_sbm(n_verts, n_blocks, B_mat_trunc)
            sbm_df = select_sbm(
                graph, n_components_try_range, n_block_try_range, directed=directed
            )
            sbm_df["n_verts"] = n_verts
            sbm_df["n_blocks"] = n_blocks
            master_sbm_df = master_sbm_df.append(sbm_df, ignore_index=True, sort=True)

    return master_sbm_df


@ex.automain
def main(
    n_sims,
    n_jobs,
    n_blocks_range,
    n_verts_range,
    n_components_try_range,
    n_block_try_range,
    B_mat,
    directed,
):
    seeds = np.random.randint(1e8, size=n_sims)

    def run(seed):
        """ Like a lambda func """
        return run_sim(
            seed,
            n_blocks_range,
            n_verts_range,
            n_components_try_range,
            n_block_try_range,
            B_mat,
            directed,
        )

    outs = Parallel(n_jobs=n_jobs, verbose=40)(delayed(run)(seed) for seed in seeds)

    columns = [
        "n_params_gmm",
        "n_params_sbm",
        "rss",
        "mse",
        "score",
        "n_components_try",
        "n_block_try",
        "n_blocks",
        "n_verts",
        "sim_ind",
    ]
    master_out_df = pd.DataFrame(columns=columns)
    for i, out in enumerate(outs):
        out["sim_ind"] = i
        master_out_df = master_out_df.append(out, ignore_index=True, sort=True)
    save_obj(master_out_df, file_obs, "master_out_df")
    return 0
