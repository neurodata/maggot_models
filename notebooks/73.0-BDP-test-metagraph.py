# %% [markdown]
# #

from src.graph import MetaGraph
import numpy as np
import pandas as pd
import networkx as nx

n_verts = 5
adj = np.arange(0, n_verts ** 2).reshape(n_verts, n_verts)
adj[0, :] = 0
adj[:, 0] = 0
print(adj)
meta = pd.DataFrame(index=range(n_verts))

meta["title"] = ["zero", "one", "two", "three", "four"]
meta["remove"] = [False, True, False, False, True]
meta.index = ["0", "1", "2", "3", "4"]
print(meta)

mg = MetaGraph(adj, meta)
mg.make_lcc()
# mg.reindex(np.array([4, 3, 1, 0]))
print(mg.meta)
print(mg.adj)

mg2 = MetaGraph(mg.g)
print(mg2.adj)
print(mg2.meta)

