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
