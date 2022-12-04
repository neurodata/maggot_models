#%%
import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

import pymaid
from graspologic.utils import symmetrize
from navis import NeuronList, TreeNeuron, nblast_allbyall
from src.data import load_maggot_graph
from src.pymaid import start_instance
from src.data import load_navis_neurons

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
mg = load_maggot_graph()
meta = mg.nodes
meta = meta[meta["selected_lcc"]]


#%% define some functions


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


neurons = load_navis_neurons()
skid_map = dict(zip([int(n.id) for n in neurons], np.arange(len(neurons))))


def neuron_lookup(neuron_ids):
    sub_neurons = []
    for neuron_id in neuron_ids:
        sub_neurons.append(neurons[skid_map[neuron_id]])
    return sub_neurons


#%%
for side in ["left", "right"]:
    print(f"Processing side: {side}")
    side_meta = meta[meta[side]]
    side_neurons = neurons.idx[side_meta.index]
    scores = nblast_allbyall(
        side_neurons,
        normalized=False,
        progress=True,
        use_alpha=False,
        smat=None,
        n_cores=1,
    )
    assert (scores.index.astype(int) == side_meta.index).all()
    assert (scores.columns.astype(int) == side_meta.index).all()
    scores = pd.DataFrame(
        data=scores.values, index=side_meta.index, columns=side_meta.index
    )
    scores.to_csv(out_dir / f"{side}-nblast-scores.csv")

    similarity = postprocess_nblast(scores)
    similarity.to_csv(out_dir / f"{side}-nblast-similarities.csv")
    print()


#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
