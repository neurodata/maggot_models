import numpy as np
from graspy.match import GraphMatch
from joblib import Parallel, delayed


def diag_indices(length, k=0):
    neg = False
    if k < 0:
        neg = True
    k = np.abs(k)
    inds = (np.arange(length - k), np.arange(k, length))
    if neg:
        return (inds[1], inds[0])
    else:
        return inds


def exp_func(k, alpha, beta=1, c=0):
    return beta * np.exp(-alpha * (k - 1)) + c


def calc_mean_by_k(ks, perm_adj):
    length = len(perm_adj)
    ps = []
    for k in ks:
        p = perm_adj[diag_indices(length, k)].mean()
        ps.append(p)
    return np.array(ps)


def get_vals_by_k(ks, perm_adj):
    ys = []
    xs = []
    for k in ks:
        y = perm_adj[diag_indices(len(perm_adj), k)]
        ys.append(y)
        x = np.full(len(y), k)
        xs.append(x)
    return np.concatenate(ys), np.concatenate(xs)


def make_flat_match(length, **kws):
    match_mat = np.zeros((length, length))
    match_mat[np.triu_indices(length, k=1)] = 1
    return match_mat


def make_linear_match(length, offset=0, **kws):
    match_mat = np.zeros((length, length))
    for k in np.arange(1, length):
        match_mat[diag_indices(length, k)] = length - k + offset
    return match_mat


def make_exp_match(adj, alpha=0.5, beta=1, c=0, norm=False, **kws):
    length = len(adj)
    match_mat = np.zeros((length, length))
    for k in np.arange(1, length):
        match_mat[diag_indices(length, k)] = exp_func(k, alpha, beta, c)
    match_mat = normalize_match(adj, match_mat, method=norm)
    return match_mat


def normalize_match(graph, match_mat, method="fro"):
    if method == "fro":
        match_mat = match_mat / np.linalg.norm(match_mat) * np.linalg.norm(graph)
    elif method == "sum":
        match_mat = match_mat / np.sum(match_mat) * np.sum(graph)
    elif method is None or method is False:
        pass
    else:
        raise ValueError("invalid method")
    return match_mat


def fit_gm_exp(
    adj,
    alpha,
    beta=1,
    c=0,
    n_init=5,
    norm=False,
    max_iter=30,
    eps=0.05,
    n_jobs=1,
    verbose=0,
    return_best=False,
):
    gm = GraphMatch(
        n_init=1, init_method="rand", max_iter=max_iter, eps=eps, shuffle_input=True
    )
    match_mat = make_exp_match(adj, alpha=alpha, beta=beta, c=c, norm=norm)

    seeds = np.random.choice(int(1e8), size=n_init)

    def _fit(seed):
        np.random.seed(seed)
        gm.fit(match_mat, adj)
        return gm.perm_inds_, gm.score_

    outs = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_fit)(s) for s in seeds)
    outs = list(zip(*outs))
    perms = np.array(outs[0])
    scores = np.array(outs[1])
    if return_best:
        ind = np.argmax(scores)
        perm = perms[ind]
        score = scores[ind]
        return perm, score
    return perms, scores


def get_best_run(perms, scores, n_opts=None):
    if n_opts is None:
        n_opts = len(perms)
    opt_inds = np.random.choice(len(perms), n_opts, replace=False)
    perms = perms[opt_inds]
    scores = scores[opt_inds]
    max_ind = np.argmax(scores)
    return perms[max_ind], scores[max_ind]
