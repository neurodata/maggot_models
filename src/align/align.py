import numpy as np
from scipy.linalg import orthogonal_procrustes


class Procrustes:
    def __init__(self, method="ortho"):
        self.method = method

    def fit(self, X, Y=None, x_seeds=None, y_seeds=None):
        if Y is None and (x_seeds is not None and y_seeds is not None):
            Y = X[y_seeds]
            X = X[x_seeds]
        elif Y is not None and (x_seeds is not None or y_seeds is not None):
            ValueError("May only use one of \{Y, \{x_seeds, y_seeds\}\}")

        X = X.copy()
        Y = Y.copy()

        if self.method == "ortho":
            R = orthogonal_procrustes(X, Y)[0]
        elif self.method == "diag-ortho":
            norm_X = np.linalg.norm(X, axis=1)
            norm_Y = np.linalg.norm(Y, axis=1)
            norm_X[norm_X <= 1e-15] = 1
            norm_Y[norm_Y <= 1e-15] = 1
            X = X / norm_X[:, None]
            Y = Y / norm_Y[:, None]
            R = orthogonal_procrustes(X, Y)[0]
        else:
            raise ValueError("Invalid `method` parameter")

        self.R_ = R
        return self

    def transform(self, X, map_inds=None):
        if map_inds is not None:
            X_transform = X.copy()
            X_transform[map_inds] = X_transform[map_inds] @ self.R_
        else:
            X_transform = X @ self.R_
        return X_transform
