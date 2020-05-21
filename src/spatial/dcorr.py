import numpy as np
from sklearn.metrics import pairwise_distances
from hyppo.ksample import KSample
from joblib import Parallel, delayed


def euclidean(x):
    """Default euclidean distance function calculation"""
    return pairwise_distances(X=x, metric="euclidean", n_jobs=-1)


def run_dcorr(data1, data2):
    ksamp = KSample("Dcorr", compute_distance=euclidean)
    stat, pval = ksamp.test(data1, data2, auto=True)
    return stat, pval


def spatial_dcorr(data1, data2, method="full", max_samples=1000, n_subsamples=5):
    if (len(data1) <= 10) or (len(data2) <= 10):
        return np.nan, np.nan

    if method == "subsample":
        if max(len(data1), len(data2)) < max_samples:
            method = "full"
        else:
            stats = np.empty(n_subsamples)
            p_vals = np.empty(n_subsamples)
            all_shuffles = []
            for i in range(n_subsamples):
                subsampled_data = []
                for data in [data1, data2]:
                    n_subsamples = min(len(data), max_samples)
                    inds = np.random.choice(
                        n_subsamples, size=n_subsamples, replace=False
                    )
                    subsampled_data.append(data[inds])
                all_shuffles.append(subsampled_data)
            outs = Parallel(n_jobs=-1)(delayed(run_dcorr)(*s) for s in all_shuffles)
            outs = list(zip(*outs))
            stats = outs[0]
            p_vals = outs[1]
            stat = np.median(stats)
            p_val = np.median(p_vals)
    if method == "max-d":
        max_dim_stat = -np.inf
        best_p_val = np.nan
        for dim in range(data1.shape[1]):
            dim_stat, dim_p_val = run_dcorr(data1[:, dim], data2[:, dim])
            if dim_stat > max_dim_stat:
                max_dim_stat = dim_stat
                best_p_val = dim_p_val
        stat = max_dim_stat
        p_val = best_p_val
    if method == "full":
        stat, p_val = run_dcorr(data1, data2)
    return stat, p_val
