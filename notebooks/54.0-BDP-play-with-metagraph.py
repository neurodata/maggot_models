# %% [markdown]
# # Imports
from src.graph import MetaGraph
from src.data import load_networkx
from graspy.utils import is_fully_connected

# %% [markdown]
# # Constants
BRAIN_VERSION = "2019-12-18"
GRAPH_TYPE = "Gad"
# %% [markdown]
# # Loads
g = load_networkx(GRAPH_TYPE, BRAIN_VERSION)
mg = MetaGraph(g)

# %% [markdown]
# # Show that getting LCC works
print(is_fully_connected(mg.g))
print(mg.n_verts)
print(mg.meta.shape)
print()
mg = mg.make_lcc()
print(is_fully_connected(mg.g))
print(mg.n_verts)
print(mg.meta.shape)

# %% [markdown]
# #
