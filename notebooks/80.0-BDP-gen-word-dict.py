# %% [markdown]
# #
import os
import urllib.request
from pathlib import Path

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graph_tool import load_graph
from graph_tool.inference import minimize_blockmodel_dl
from joblib import Parallel, delayed
from random_word import RandomWords
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid

from graspy.utils import cartprod
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


def random_names(space=False):
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
    return name_words
    # names = []
    # for i in range(n_names):

    #     rand_name = base_char.join(
    #         [name_words[np.random.randint(0, len(name_words))] for i in range(2)]
    #     )
    #     names.append(rand_name)
    # return np.array(names)


name_words = random_names()
name_df = pd.DataFrame(name_words)
name_df.to_csv("./maggot_models/src/visualization/names.csv", header=False, index=False)


# %%
from src.visualization import NAMES, random_names

NAMES


#%%
random_names(10)

# %%
