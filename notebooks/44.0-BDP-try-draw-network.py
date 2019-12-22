# %% [markdown]
# #
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import networkx as nx
import numpy as np
from src.hierarchy import signal_flow
from graspy.models import SBMEstimator

node_signal_flow = signal_flow(adj)
mean_sf = np.zeros(k)
for i in np.unique(pred_labels):
    inds = np.where(pred_labels == i)[0]
    mean_sf[i] = np.mean(node_signal_flow[inds])

cluster_mean_latent = gmm.model_.means_[:, 0]
block_probs = SBMEstimator().fit(bin_adj, y=pred_labels).block_p_
block_prob_df = pd.DataFrame(data=block_probs, index=range(k), columns=range(k))
block_g = nx.from_pandas_adjacency(block_prob_df, create_using=nx.DiGraph)
plt.figure(figsize=(10, 10))
# don't ever let em tell you you're too pythonic
pos = dict(zip(range(k), zip(cluster_mean_latent, mean_sf)))
# nx.draw_networkx_nodes(block_g, pos=pos)
labels = nx.get_edge_attributes(block_g, "weight")

# nx.draw_networkx_edge_labels(block_g, pos, edge_labels=labels)

norm = mpl.colors.LogNorm(vmin=0.01, vmax=0.1)

sm = ScalarMappable(cmap="Reds", norm=norm)
cmap = sm.to_rgba(np.array(list(labels.values())) + 0.01)
nx.draw_networkx(
    block_g,
    pos,
    edge_cmap="Reds",
    edge_color=cmap,
    connectionstyle="arc3,rad=0.2",
    width=1.5,
)
