import numpy as np
from graspy.utils import remove_loops


def signal_flow(A, n_components=5, return_evals=False):
    """Implementation of the signal flow metric from Varshney et al 2011
    
    Parameters
    ----------
    A : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    A = A.copy()
    A = remove_loops(A)
    W = (A + A.T) / 2

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    b = np.sum(W * np.sign(A - A.T), axis=1)
    L_pinv = np.linalg.pinv(L)
    z = L_pinv @ b

    return z


def normalized_laplacian(A, n_components=5, return_evals=False, normalize_evecs=True):
    W = (A + A.T) / 2

    D = np.diag(np.sum(W, axis=1))

    L = D - W
    D_root = np.diag(np.diag(D) ** (-1 / 2))
    D_root[np.isnan(D_root)] = 0
    D_root[np.isinf(D_root)] = 0
    Q = D_root @ L @ D_root
    evals, evecs = np.linalg.eig(Q)
    inds = np.argsort(evals)
    evals = evals[inds]
    evecs = evecs[:, inds]
    if normalize_evecs:
        evecs = np.diag(np.diag(D) ** (-1 / 2)) @ evecs
    # print(evecs[:, 0])
    if return_evals:
        return evecs[:, :n_components], evals[:n_components]
    else:
        return evecs[:, n_components]
    # scatter_df = pd.DataFrame()
    # for i in range(1, n_components + 1):
    #     scatter_df[f"Lap-{i+1}"] = evecs[:, i]
    # scatter_df["Signal flow"] = z
    # if return_evals:
    #     return scatter_df, evals
    # else:
    #     return scatter_df
