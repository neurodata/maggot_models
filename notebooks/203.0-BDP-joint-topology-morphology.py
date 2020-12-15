#%%

from src.data import load_metagraph

mg = load_metagraph("G")
meta = mg.meta
meta = meta[meta["left"]]
class1 = [
    "KC",
    "MBIN",
    "MBON",
    "uPN",
    "tPN",
    "vPN",
    "bLN",
    "pLN",
    "APL",
]
class2 = ["ORN"]
meta = meta[meta["class1"].isin(class1) | meta["class2"].isin(class2)]
meta["merge_class"].unique()
mg = mg.reindex(meta.index, use_ids=True)
mg = mg.remove_pdiff()
mg = mg.make_lcc()
meta = mg.meta
print(len(meta))
#%%

import os

os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources/"
os.environ[
    "R_USER"
] = "/Users/bpedigo/miniconda3/envs/maggot_graspologic/lib/python3.7/site-packages/rpy2"

# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr

# importr("nat")
# import rpy2.robjects.numpy2ri

# rpy2.robjects.numpy2ri.activate()
# epca = importr("epca")
# return epca


from src.pymaid import start_instance

start_instance()

import pymaid
from navis import TreeNeuron, NeuronList


#%%
# meta = meta.iloc[:100]
neuron_ids = meta.index
print(len(neuron_ids))
neuron_ids = [int(n) for n in neuron_ids]
neurons = pymaid.get_neuron(neuron_ids)
treenode_tables = []
for neuron_id, neuron in zip(neuron_ids, neurons):
    treenode_table = pymaid.get_treenode_table(neuron, include_details=False)
    treenode_tables.append(treenode_table)

success_neurons = []
tree_neurons = []
for neuron_id, treenode_table in zip(neuron_ids, treenode_tables):
    treenode_table.rename(columns={"parent_node_id": "parent_id"}, inplace=True)

    tree_neuron = TreeNeuron(treenode_table)
    if (tree_neuron.soma is not None) and (len(tree_neuron.soma) > 1):
        print(f"Neuron {neuron_id} has more than one soma, removing")
        print(meta.loc[neuron_id])
    else:
        tree_neurons.append(tree_neuron)
        success_neurons.append(neuron_id)

tree_neurons = NeuronList(tree_neurons)
meta = meta.loc[success_neurons]

#%%
from navis.interfaces import r as natr

output = natr.nblast_allbyall(tree_neurons, n_cores=1, normalized=False)
print(output)
import seaborn as sns
import matplotlib.pyplot as plt
from src.visualization import adjplot, CLASS_COLOR_DICT
from graspologic.utils import symmetrize
from src.io import savefig

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


similarity = output.values
sym_similarity = symmetrize(similarity)
adjplot(
    sym_similarity,
    sort_class=meta["class1"],
    colors=meta["class1"],
    palette=CLASS_COLOR_DICT,
    tick_rot=45,
    cbar_kws=dict(shrink=0.7),
)
stashfig("sym_similarity")
plt.show()

# stashfig()