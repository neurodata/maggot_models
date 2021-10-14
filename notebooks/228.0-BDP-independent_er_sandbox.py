#%%
import numpy as np
from graspologic.simulations import er_np

p1 = 0.05
p2 = 0.03
p3 = 0.02
p4 = 0.01
ps = [p1, p2, p3, p4]
n = 100

As = np.zeros((4, n, n))
for i, p in enumerate(ps):
    As[i] = er_np(n, p, directed=True, loops=True)

As.sum(axis=0)