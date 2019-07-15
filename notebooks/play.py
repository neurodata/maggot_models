#%%
from src.data import load_left
from graspy.models import DCSBMEstimator

graph, labels = load_left()

dcsbm = DCSBMEstimator(directed=True, loops=False, degree_directed=True)
dcsbm.fit(graph, y=labels)
dcsbm.mse(graph)
#%%
