setup = """
from hyppo.ksample import KSample
from sklearn.datasets import make_blobs

n_samples = 1000
X1, y1 = make_blobs(n_samples=n_samples, n_features=3, centers=10)
X2, y2 = make_blobs(n_samples=n_samples, n_features=3, centers=9)


def broadcast_dcorr(data1, data2):
    ksamp = KSample("Dcorr", bias=False)
    stat, pval = ksamp.test(data1, data2, auto=True, workers=-1)
    return stat, pval


def normal_dcorr(data1, data2):
    ksamp = KSample("Dcorr", bias=True)
    stat, pval = ksamp.test(data1, data2, auto=True, workers=-1)
    return stat, pval
"""
print(setup)

from timeit import timeit

timeit("broadcast_dcorr(X1, X2)", number=2, setup=setup)
