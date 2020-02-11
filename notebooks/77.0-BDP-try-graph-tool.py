# %% [markdown]
# #
import os

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import urllib.request
from graph_tool import load_graph
from graph_tool.inference import minimize_blockmodel_dl
from joblib import Parallel, delayed
from random_word import RandomWords
from sklearn.model_selection import ParameterGrid

from src.data import load_metagraph
from src.graph import MetaGraph
from src.io import savecsv, savefig
from src.utils import get_blockmodel_df
from src.visualization import (
    CLASS_COLOR_DICT,
    CLASS_IND_DICT,
    barplot_text,
    probplot,
    remove_spines,
    stacked_barplot,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def random_names(n_names=1, space=False):
    # modified from
    # https://stackoverflow.com/questions/18834636/random-word-generator-python
    word_url = (
        "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type"
        + "=text/plain"
    )
    response = None
    n_attempts = 0
    while response is None:
        if n_attempts > 100:
            raise Exception("this random word shit broke")
        try:
            response = urllib.request.urlopen(word_url, timeout=1)
        except:
            n_attempts += 1
            pass

    if space:
        base_char = " "
    else:
        base_char = ""

    long_txt = None
    n_attempts = 0
    while long_txt is None:
        if n_attempts > 100:
            raise Exception("this random word shit broke 2")
        try:
            long_txt = response.read().decode()
        except:
            n_attempts += 1
            pass

    words = long_txt.splitlines()
    upper_words = [word for word in words if word[0].isupper()]
    name_words = [word for word in upper_words if not word.isupper()]
    names = []
    for i in range(n_names):

        rand_name = base_char.join(
            [name_words[np.random.randint(0, len(name_words))] for i in range(2)]
        )
        names.append(rand_name)
    return np.array(names)


def get_random_word():
    r = RandomWords()
    n_attempts = 0
    word = None
    while word is None:
        if n_attempts > 1000:
            raise Exception("this random word shit broke")
        try:
            word = r.get_random_word(includePartOfSpeech="noun")
        except:
            n_attempts += 1
            pass
    return word


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)
    plt.close()


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


def augment_classes(skeleton_labels, class_labels, lineage_labels, fill_unk=True):
    if fill_unk:
        classlin_labels = class_labels.copy()
        fill_inds = np.where(class_labels == "unk")[0]
        classlin_labels[fill_inds] = lineage_labels[fill_inds]
        used_inds = np.array(list(CLASS_IND_DICT.values()))
        unused_inds = np.setdiff1d(range(len(cc.glasbey_light)), used_inds)
        lineage_color_dict = dict(
            zip(np.unique(lineage_labels), np.array(cc.glasbey_light)[unused_inds])
        )
        color_dict = {**CLASS_COLOR_DICT, **lineage_color_dict}
        hatch_dict = {}
        for key, val in color_dict.items():
            if key[0] == "~":
                hatch_dict[key] = "//"
            else:
                hatch_dict[key] = ""
    else:
        color_dict = "class"
        hatch_dict = None
    return classlin_labels, color_dict, hatch_dict


def run_minimize_blockmodel(mg, temp_loc):
    # save to temp
    nx.write_graphml(mg.g, temp_loc)
    # load into graph-tool from temp
    g = load_graph(temp_loc, fmt="graphml")
    total_degrees = g.get_total_degrees(g.get_vertices())
    remove_verts = np.where(total_degrees == 0)[0]
    g.remove_vertex(remove_verts)
    min_state = minimize_blockmodel_dl(g, verbose=False)

    blocks = list(min_state.get_blocks())
    verts = g.get_vertices()

    block_map = {}

    for v, b in zip(verts, blocks):
        cell_id = int(g.vertex_properties["_graphml_vertex_id"][v])
        block_map[cell_id] = int(b)

    block_series = pd.Series(block_map)
    block_series.name = "block_label"
    return block_series


VERSION = "2020-01-29"
BLIND = True

# parameters
# TODO weights
# TODO sym thresh or not


def run_experiment(seed=None, graph_type=None, threshold=None, param_key=None):
    np.random.seed(seed)
    if BLIND:
        temp_param_key = param_key.replace(" ", "")  # don't want spaces in filenames
        savename = f"{temp_param_key}-cell-types-"
        title = param_key
    else:
        savename = f"{graph_type}-t{threshold}-cell-types"
        title = f"{graph_type}, threshold = {threshold}"

    mg = load_metagraph(graph_type, version=VERSION)

    # simple threshold
    # TODO they will want symmetric threshold...
    # TODO maybe make that a parameter
    adj = mg.adj.copy()
    adj[adj <= threshold] = 0
    meta = mg.meta.copy()
    meta = pd.DataFrame(mg.meta["neuron_name"])
    mg = MetaGraph(adj, meta)

    # run the graphtool code
    temp_loc = f"maggot_models/data/interim/temp-{param_key}.graphml"
    block_series = run_minimize_blockmodel(mg, temp_loc)

    # manage the output
    mg = load_metagraph(graph_type, version=VERSION)
    mg.meta = pd.concat((mg.meta, block_series), axis=1)
    mg.meta["Original index"] = range(len(mg.meta))
    keep_inds = mg.meta[~mg.meta["block_label"].isna()]["Original index"].values
    mg.reindex(keep_inds)
    if graph_type != "G":
        mg.verify(10000, graph_type=graph_type, version=VERSION)

    # deal with class labels
    lineage_labels = mg.meta["lineage"].values
    lineage_labels = np.vectorize(lambda x: "~" + x)(lineage_labels)
    class_labels = mg["Merge Class"]
    skeleton_labels = mg.meta.index.values
    classlin_labels, color_dict, hatch_dict = augment_classes(
        skeleton_labels, class_labels, lineage_labels
    )
    block_label = mg["block_label"].astype(int)

    # barplot with unknown class labels merged in, proportions
    _, _, order = barplot_text(
        block_label,
        classlin_labels,
        norm_bar_width=True,
        color_dict=color_dict,
        hatch_dict=hatch_dict,
        title=title,
        figsize=(24, 18),
        return_order=True,
    )
    stashfig(savename + "barplot-mergeclasslin-props")
    category_order = np.unique(block_label)[order]

    # barplot with regular class labels
    barplot_text(
        block_label,
        class_labels,
        norm_bar_width=True,
        color_dict=color_dict,
        hatch_dict=hatch_dict,
        title=title,
        figsize=(24, 18),
        category_order=category_order,
    )
    stashfig(savename + "barplot-mergeclass-props")

    # barplot with unknown class labels merged in, counts
    barplot_text(
        block_label,
        classlin_labels,
        norm_bar_width=False,
        color_dict=color_dict,
        hatch_dict=hatch_dict,
        title=title,
        figsize=(24, 18),
        return_order=True,
        category_order=category_order,
    )
    stashfig(savename + "barplot-mergeclasslin-counts")

    # barplot of hemisphere membership
    fig, ax = plt.subplots(1, 1, figsize=(10, 20))
    stacked_barplot(
        block_label,
        mg["Hemisphere"],
        norm_bar_width=True,
        category_order=category_order,
        ax=ax,
    )
    remove_spines(ax)
    stashfig(savename + "barplot-hemisphere")

    # plot block probability matrix
    counts = False
    weights = False
    prob_df = get_blockmodel_df(
        mg.adj, block_label, return_counts=counts, use_weights=weights
    )
    prob_df = prob_df.reindex(order, axis=0)
    prob_df = prob_df.reindex(order, axis=1)
    ax = probplot(
        100 * prob_df, fmt="2.0f", figsize=(20, 20), title=title, font_scale=0.4
    )
    stashfig(savename + "probplot")
    block_series.name = param_key
    return block_series


np.random.seed(8888)
n_replicates = 3
param_grid = {"graph_type": ["Gad"], "threshold": [0, 1, 2, 3, 4]}
params = list(ParameterGrid(param_grid))
seeds = np.random.randint(1e8, size=n_replicates * len(params))
param_keys = random_names(len(seeds))

rep_params = []
for i, seed in enumerate(seeds):
    p = params[i % len(params)].copy()
    p["seed"] = seed
    p["param_key"] = param_keys[i]
    rep_params.append(p)

print(rep_params)
# %% [markdown]
# #
outs = Parallel(n_jobs=-2, verbose=10)(delayed(run_experiment)(**p) for p in rep_params)

# %% [markdown]
# #
block_df = pd.concat(outs, axis=1, ignore_index=False)
stashcsv(block_df, "block-labels")
param_df = pd.DataFrame(rep_params)
stashcsv(param_df, "parameters")

