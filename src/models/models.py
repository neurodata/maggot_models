import numpy as np
import pandas as pd

# def estimate_sbm(
#     graph,
#     n_communities,
#     n_components=None,
#     directed=False,
#     method="gc",
#     metric=None,
#     rank="full",
# ):
#     if n_communities == 1:
#         estimator = EREstimator(directed=directed, loops=False)
#         estimator.fit(graph)
#         n_params = estimator._n_parameters()
#     else:
#         vertex_assignments, n_params = estimate_assignments(
#             graph, n_communities, n_components, method=method, metric=metric
#         )
#         estimator = SBMEstimator(directed=directed, loops=False, rank=rank)
#         estimator.fit(graph, y=vertex_assignments)
#     return estimator, n_params
from sklearn.metrics import make_scorer, mean_squared_error


from graspy.cluster import GaussianCluster
from graspy.embed import AdjacencySpectralEmbed
from graspy.models import DCSBMEstimator, EREstimator, RDPGEstimator, SBMEstimator
from graspy.utils import is_symmetric

from ..utils import compute_mse, compute_rss
from .grid_search import GridSearchUS
from .brute_cluster import brute_cluster


def estimate_assignments(
    graph, n_communities, n_components=None, method="gc", metric=None
):
    """Given a graph and n_comunities, sweeps over covariance structures
    Not deterministic
    Not using graph bic or mse to calculate best

    1. Does an embedding on the raw graph
    2. GaussianCluster on the embedding. This will sweep covariance structure for the 
       given n_communities
    3. Returns n_parameters based on the number used in GaussianCluster

    method can be "gc" or "bc" 

    method 
    "gc" : use graspy GaussianCluster
        this defaults to full covariance
    "bc" : tommyclust with defaults
        so sweep covariance, agglom, linkage
    "bc-metric" : tommyclust with custom metric
        still sweep everything
    "bc-none" : mostly for testing, should behave just like GaussianCluster

    """
    embed_graph = graph.copy()
    latent = AdjacencySpectralEmbed(n_components=n_components).fit_transform(
        embed_graph
    )
    if isinstance(latent, tuple):
        latent = np.concatenate(latent, axis=1)
    if method == "gc":
        gc = GaussianCluster(
            min_components=n_communities,
            max_components=n_communities,
            covariance_type="all",
        )
        vertex_assignments = gc.fit_predict(latent)
        n_params = gc.model_._n_parameters()
    elif method == "bc":
        vertex_assignments, n_params = brute_cluster(latent, [n_communities])
    elif method == "bc-metric":
        vertex_assignments, n_params = brute_cluster(
            latent, [n_communities], metric=metric
        )
    elif method == "bc-none":
        vertex_assignments, n_params = brute_cluster(
            latent,
            [n_communities],
            affinities=["none"],
            linkages=["none"],
            covariance_types=["full"],
        )
    else:
        raise ValueError("Unspecified clustering method")
    return (vertex_assignments, n_params)


def estimate_rdpg(graph, n_components=None):
    estimator = RDPGEstimator(loops=False, n_components=n_components)
    estimator.fit(graph)
    if n_components is None:
        n_components = estimator.latent_.shape[0]
    # n_params = graph.shape[0] * n_components
    n_params = estimator._n_parameters()
    return estimator, n_params


def select_sbm(
    graph,
    param_grid,
    directed=True,
    co_block=False,
    metric="mse",
    c=0,
    rank="full",
    n_jobs=1,
    n_init=1,
):
    """sweeps over n_components, n_blocks, fits an sbm for each 
    Using GaussianCluster, so will internally sweep covariance structure and pick best

    Returns n_params for the gaussian
    N_params for the sbm kinda
    rss
    score

    Maybe at some point this will sweep rank of B

    Parameters
    ----------
    graph : [type]
        [description]
    n_block_try_range : [type]
        [description]
    n_components_try_range : [type]
        [description]
    directed : bool, optional
        [description], by default False
    """

    # common parameters of all estimators
    sbm = SBMEstimator(
        directed=directed,
        loops=False,
        n_init=n_init,
        co_block=co_block,
        metric=metric,
        rank=rank,
    )

    # define scoring functions to evaluate models
    scorers = gen_scorers(sbm, graph)

    # run the grid search
    grid_search = GridSearchUS(
        sbm, param_grid, scoring=scorers, n_jobs=n_jobs, verbose=5, refit=False
    )
    grid_search.fit(graph)

    out_df = grid_search.cv_results_
    # out_df["param_regularizer"] = [
    #     v["regularizer"] for v in out_df["param_embed_kws"].values
    # ]
    # out_dict = {}
    # for i, n_components_try in enumerate(n_components_try_range):
    #     for j, n_block_try in enumerate(n_block_try_range):
    #         # check special case for ER, don't need to cluster
    #         if n_block_try == 1:
    #             vertex_assignments = np.zeros(graph.shape[0])
    #             n_params_gmm = 1
    #         else:
    #             vertex_assignments, n_params_gmm = estimate_assignments(
    #                 graph, n_block_try, n_components_try, method=method, metric=metric
    #             )

    #         if rank == "sweep":
    #             rank_try_range = list(range(1, n_block_try + 1))
    #         else:
    #             rank_try_range = [n_block_try]

    #         for k, rank_try in enumerate(rank_try_range):
    #             ind = i * len(n_block_try_range) + j * len(rank_try_range) + k

    #             estimator = SBMEstimator(directed=directed, loops=False, rank=rank_try)
    #             estimator.fit(graph, y=vertex_assignments)

    #             rss = compute_rss(estimator, graph)
    #             mse = compute_mse(estimator, graph)
    #             score = np.sum(estimator.score_samples(graph, clip=c))
    #             n_params_sbm = estimator._n_parameters()
    #             # account for the estimated positions
    #             if type(estimator) == SBMEstimator:
    #                 n_params_sbm += estimator.block_p_.shape[0] - 1

    #             out_dict[ind] = {
    #                 "n_params_gmm": n_params_gmm,
    #                 "n_params_sbm": n_params_sbm,
    #                 "rss": rss,
    #                 "mse": mse,
    #                 "score": score,
    #                 "n_components_try": n_components_try,
    #                 "n_block_try": n_block_try,
    #                 "rank_try": rank_try,
    #             }
    # out_df = pd.DataFrame.from_dict(out_dict, orient="index")
    return out_df


def select_dcsbm(
    graph,
    param_grid,
    directed=True,
    degree_directed=False,
    metric="mse",
    c=0,
    rank="full",
    n_jobs=1,
    n_init=1,
):
    # common parameters of all estimators
    dcsbm = DCSBMEstimator(
        directed=directed, degree_directed=degree_directed, loops=False, metric=metric
    )

    # define scoring functions to evaluate models
    scorers = gen_scorers(dcsbm, graph)

    # run the grid search
    grid_search = GridSearchUS(
        dcsbm,
        param_grid,
        scoring=scorers,
        n_jobs=n_jobs,
        verbose=0,
        refit=False,
        n_init=n_init,
    )
    grid_search.fit(graph)

    # format outputs
    out_df = grid_search.cv_results_
    out_df["param_regularizer"] = [
        v["regularizer"] for v in out_df["param_embed_kws"].values
    ]

    return out_df


def select_rdpg(graph, param_grid, directed=True, n_jobs=1):
    if is_symmetric(graph) and directed:
        msg = (
            "select_RDPG was given an undirected graph but you wanted"
            + " a directed model"
        )
        raise ValueError(msg)

    rdpg = RDPGEstimator(loops=False)

    # # define scoring functions to evaluate models
    scorers = gen_scorers(rdpg, graph)

    # run the grid search
    grid_search = GridSearchUS(
        rdpg, param_grid, scoring=scorers, n_jobs=n_jobs, verbose=0
    )
    grid_search.fit(graph)

    return grid_search.cv_results_


def gen_scorers(estimator, graph):
    small_clip = 1 / (graph.size - graph.shape[0])

    def n_params_scorer(estimator, graph, y=None):
        return estimator._n_parameters()

    def mse_scorer(estimator, graph, y=None):
        return estimator.mse(graph)

    def zeroclip_mse_scorer(estimator, graph, y=None):
        return estimator.mse(graph, clip=0)

    def smallclip_mse_scorer(estimator, graph, y=None):
        return estimator.mse(graph, clip=small_clip)

    def likelihood_scorer(estimator, graph, y=None):
        return estimator.score(graph)

    def zeroclip_likelihood_scorer(estimator, graph, y=None):
        return estimator.score(graph, clip=0)

    def smallclip_likelihood_scorer(estimator, graph, y=None):
        return estimator.score(graph, clip=small_clip)

    scorers = {
        "n_params": n_params_scorer,
        "mse": mse_scorer,
        "zc_mse": zeroclip_mse_scorer,
        "sc_mse": smallclip_mse_scorer,
        "likelihood": likelihood_scorer,
        "zc_likelihood": zeroclip_likelihood_scorer,
        "sc_likelihood": smallclip_likelihood_scorer,
    }
    return scorers


def fit_a_priori(estimator, graph, labels):
    param_grid = {}
    scorers = gen_scorers(estimator, graph)
    grid_search = GridSearchUS(estimator, param_grid, scoring=scorers, verbose=0)
    grid_search.fit(graph, y=labels)
    return grid_search.cv_results_
