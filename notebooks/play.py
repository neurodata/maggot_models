#%%
from src.data import load_left
from graspy.models import DCSBMEstimator

graph, labels = load_left()

dcsbm = DCSBMEstimator(directed=True, loops=False, degree_directed=True)
dcsbm.fit(graph, y=labels)
dcsbm.mse(graph)
#%%
from src.models import GridSearchUS
from src.models import select_rdpg

n_init = 3
n_components_try = range(1, 5)
param_grid = dict(n_components=n_components_try)
select_rdpg(graph, param_grid)


#%%
from graspy.utils import cartprod
import numpy as np

s = range(20, 25)
f = np.random.uniform(size=5)

out = cartprod(s, f)
from itertools import product

out = product(s, f)
for i, j in product(s, f):
    print(i)
    print(j)
    print()
#%%
for i, j in out:
    print(i)
    print(j)
    print()

#%%
out

#%%
from src.models import GridSweep

from src.models import gen_scorers

estimator = DCSBMEstimator
scorers = gen_scorers(estimator, graph)

param_grid = dict(n_blocks=list(range(1, 5)), degree_directed=(True, False))
gs = GridSweep(
    estimator, param_grid, scoring=scorers, n_init=10, refit="mse", verbose=5
)
gs.fit(graph)
gs.result_df_
gs.model_
#%%
