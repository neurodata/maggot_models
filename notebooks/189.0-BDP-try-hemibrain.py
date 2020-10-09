# %%
import pandas as pd
import networkx as nx

meta_loc = "maggot_models/data/raw/exported-traced-adjacencies-v1.1/traced-neurons.csv"
adj_loc = "maggot_models/data/raw/exported-traced-adjacencies-v1.1/traced-total-connections.csv"
meta = pd.read_csv(meta_loc, index_col=0)
g = nx.read_weighted_edgelist(
    adj_loc, delimiter=",", create_using=nx.DiGraph, nodetype=int
)
print(len(g))

# %% [markdown]
# ##

from graspologic.embed import LaplacianSpectralEmbed
from graspologic.utils import pass_to_ranks
from graspologic.plot import pairplot


nodelist = list(sorted(g.nodes))
adj = nx.to_numpy_array(g, nodelist=nodelist)

lse = LaplacianSpectralEmbed(form="R-DAD", n_components=16)
latent_left, latent_right = lse.fit_transform(pass_to_ranks(adj))


from factor_analyzer import Rotator

rot_latent_left = Rotator(normalize=False).fit_transform(latent_left)
pairplot(rot_latent_left[:, :16])
