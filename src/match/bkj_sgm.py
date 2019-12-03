import numpy as np
from scipy.sparse import csr_matrix

from sgm import JVSparseSGM, ScipyJVClassicSGM
from graspy.match import SinkhornKnopp


def doubly_stochastic(n, barycenter=False):
    sk = SinkhornKnopp()
    K = np.random.rand(
        n, n
    )  # generate a nxn matrix where each entry is a random integer [0,1]
    for i in range(10):  # perform 10 iterations of Sinkhorn balancing
        K = sk.fit(K)
    if barycenter:
        J = np.ones((n, n)) / float(n)  # initialize J, a doubly stochastic barycenter
        P = (K + J) / 2
    else:
        P = K
    return P


class GraphMatch:
    def __init__(self, solver="normal", n_init=10, n_iter=100, tol=10, verbose=False):
        self.solver = solver
        self.n_init = n_init
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, A, B):
        n_verts = A.shape[0]
        best_score = np.inf
        best_B_out = None
        best_perm_inds = None
        for i in range(self.n_init):
            # print(i)
            P_init = doubly_stochastic(n_verts, barycenter=True)
            if self.solver == "normal":
                solver = ScipyJVClassicSGM
            if self.solver == "sparse":
                solver = JVSparseSGM
            faq = solver(
                A=csr_matrix(A), B=csr_matrix(B), P=csr_matrix(P_init), verbose=False
            )
            node_map = faq.run(self.n_iter, self.tol, self.verbose)

            P_out = csr_matrix((np.ones(n_verts), (np.arange(n_verts), node_map)))
            B_out = P_out @ B @ P_out

            score = np.abs(A - B_out).sum()

            # print(score)
            if score < best_score:
                best_score = score
                best_B_out = B_out
                best_perm_inds = node_map

            if score == 0:
                break

        self.B_out_ = best_B_out
        self.perm_inds_ = best_perm_inds
        return self

    def fit_predict(self, A, B):
        self = self.fit(A, B)
        return self.B_out_, self.perm_inds_
