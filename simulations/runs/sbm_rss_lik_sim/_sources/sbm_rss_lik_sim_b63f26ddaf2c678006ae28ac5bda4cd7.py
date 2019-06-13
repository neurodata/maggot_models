from sacred import Experiment
from sacred.observers import SlackObserver, FileStorageObserver
from src.utils import estimate_sbm, compute_rss, compute_log_lik, gen_B, gen_sbm
from joblib import wrap_non_picklable_objects
from os.path import basename
import numpy as np

from joblib import Parallel, delayed

ex = Experiment("SBM model selection")

slack_obs = SlackObserver.from_config("slack.json")
f = basename(__file__)[:-3]
file_obs = FileStorageObserver.create(
    f"/Users/bpedigo/JHU_code/maggot_models/maggot_models/simulations/runs/{f}"
)
ex.observers.append(slack_obs)
ex.observers.append(file_obs)


@ex.config
def my_config1():
    n_sims = 2  # noqa: F841
    n_blocks_range = [1, 2, 3, 4, 5, 6, 7, 8]
    n_verts_range = [100, 200, 300, 500, 1000]  # noqa: F841
    # n_jobs = 50  # noqa: F841
    n_block_try_range = list(range(1, 12))  # noqa: F841

    # keep these the same
    a = 0.1
    b = 0.1
    assortivity = 6  # how strong to make the block diagonal

    B_mat = gen_B(n_blocks_range[-1], a=a, b=b, assortivity=assortivity)  # noqa: F841
    # B_mat = B_mat.tolist()
    # let Zhu/Ghodsie choose embedding dimension
    n_components = None  # noqa: F841


@delayed
@wrap_non_picklable_objects
@ex.capture
def run_sim(
    seed, n_blocks_range, n_verts_range, n_block_try_range, B_mat, n_components
):
    np.random.seed(seed)
    sbm_score = np.zeros(
        (len(n_blocks_range), len(n_verts_range), len(n_block_try_range))
    )
    sbm_n_params = np.zeros_like(sbm_score)
    sbm_rss = np.zeros_like(sbm_score)

    for j, n_blocks in enumerate(n_blocks_range):
        B_mat_trunc = B_mat[:n_blocks, :n_blocks]
        for k, n_verts in enumerate((n_verts_range)):
            graph, labels = gen_sbm(n_verts, n_blocks, B_mat_trunc)
            for l, n_block_try in enumerate(n_block_try_range):
                estimator, n_params = estimate_sbm(
                    graph, n_block_try, n_components=n_components, directed=False
                )

                rss = compute_rss(estimator, graph)
                score = compute_log_lik(estimator, graph)
                sbm_rss[j, k, l] = rss
                sbm_score[j, k, l] = score
                sbm_n_params[j, k, l] = n_params

    out = {"sbm": {"rss": sbm_rss, "score": sbm_score, "n_params": sbm_n_params}}
    return out


@ex.automain
def main(n_sims, n_blocks_range, n_verts_range, n_block_try_range, B_mat, n_components):
    seeds = np.random.randint(1e8, size=n_sims)

    def run(seed):
        # n_jobs -2 uses all but one cores
        return run_sim(
            seed, n_blocks_range, n_verts_range, n_block_try_range, B_mat, n_components
        )

    outs = Parallel(n_jobs=1, verbose=40)(delayed(run)(seed) for seed in seeds)
    return outs

