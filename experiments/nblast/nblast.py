#%%
import logging
import time

import numpy as np
import pandas as pd
import pymaid
from sklearn.preprocessing import QuantileTransformer
from pathlib import Path

from graspologic.utils import symmetrize
from navis import NeuronList, TreeNeuron, nblast_allbyall
from src.data import load_metagraph
from src.pymaid import start_instance

# REF: https://stackoverflow.com/questions/35326814/change-level-logged-to-ipython-jupyter-notebook
# logger = logging.getLogger()
# # assert len(logger.handlers) == 1
# handler = logger.handlers[0]
# handler.setLevel(logging.ERROR)

t0 = time.time()

# for pymaid to pull neurons
start_instance()

out_dir = Path("maggot_models/experiments/nblast/outs")

#%% load connectivity data
mg = load_metagraph("G")
meta = mg.meta

#%% define some functions


def pairwise_nblast(neuron_ids, point_thresh=20):
    neuron_ids = [int(n) for n in neuron_ids]
    neurons = pymaid.get_neuron(neuron_ids)  # load in with pymaid

    # HACK: I am guessing there is a better way to do the below?
    # TODO: I was also getting some errors about neurons with more that one soma, so I
    # threw them out for now.
    treenode_tables = []
    for neuron_id, neuron in zip(neuron_ids, neurons):
        treenode_table = pymaid.get_node_table(neuron, include_details=False)
        treenode_tables.append(treenode_table)

    success_neurons = []
    tree_neurons = []
    for neuron_id, treenode_table in zip(neuron_ids, treenode_tables):
        treenode_table.rename(columns={"parent_node_id": "parent_id"}, inplace=True)

        tree_neuron = TreeNeuron(treenode_table)
        if (tree_neuron.soma is not None) and (len(tree_neuron.soma) > 1):
            print(f"Neuron {neuron_id} has more than one soma, removing")
        elif len(treenode_table) < point_thresh:
            print(f"Neuron {neuron_id} has fewer than {point_thresh} points, removing")
        else:
            tree_neurons.append(tree_neuron)
            success_neurons.append(neuron_id)

    tree_neurons = NeuronList(tree_neurons)
    print(f"{len(tree_neurons)} neurons ready for NBLAST")

    currtime = time.time()
    # NOTE: I've had too modify original code to allow smat=None
    # NOTE: this only works when normalized=False also
    scores = nblast_allbyall(
        tree_neurons, normalized=False, progress=True, use_alpha=False
    )
    print(f"{time.time() - currtime:.3f} elapsed to run NBLAST.")

    scores = pd.DataFrame(
        data=scores.values, index=success_neurons, columns=success_neurons
    )

    return scores


def postprocess_nblast(scores):
    distance = scores.values  # the raw nblast scores are dissimilarities/distances
    sym_distance = symmetrize(distance)  # the raw scores are not symmetric
    # make the distances between 0 and 1
    sym_distance /= sym_distance.max()
    sym_distance -= sym_distance.min()
    # and then convert to similarity
    morph_sim = 1 - sym_distance

    # rank transform the similarities
    # NOTE this is very different from what native NBLAST does and could likely be
    # improved upon a lot. I did this becuase it seemed like a quick way of accounting
    # for difference in scale for different neurons as well as the fact that the raw
    # distribution of similaritys was skewed low (very few small values)
    quant = QuantileTransformer()
    indices = np.triu_indices_from(morph_sim, k=1)
    transformed_vals = quant.fit_transform(morph_sim[indices].reshape(-1, 1))
    transformed_vals = np.squeeze(transformed_vals)
    # this is a discrete version of PTR basically
    ptr_morph_sim = np.ones_like(morph_sim)
    ptr_morph_sim[indices] = transformed_vals
    ptr_morph_sim[indices[::-1]] = transformed_vals

    ptr_morph_sim = pd.DataFrame(
        data=ptr_morph_sim, index=scores.index, columns=scores.columns
    )

    return ptr_morph_sim


#%% run nblast
for side in ["left", "right"]:
    print(f"Processing side: {side}")
    side_meta = meta[meta[side]]

    scores = pairwise_nblast(side_meta.index.values)
    scores.to_csv(out_dir / f"{side}-nblast-scores.csv")

    similarity = postprocess_nblast(scores)
    similarity.to_csv(out_dir / f"{side}-nblast-similarities.csv")
    print()

#%%
print("\n\n")
print(f"{time.time() - t0:.3f} elapsed for whole script.")