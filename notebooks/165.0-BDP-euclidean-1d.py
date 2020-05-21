import numpy as np


x = np.random.rand(100)
y = np.random.rand(100)

sort_inds = np.argsort(x)
x = x[sort_inds]
y = y[sort_inds]
