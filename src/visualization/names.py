import pandas as pd
import os
import numpy as np

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "names.csv")
names_csv = pd.read_csv(path, header=None, squeeze=True)
NAMES = np.array(names_csv.values).astype(str)


def random_names(n_names=1, sep=""):
    names = []
    for i in range(n_names):
        name = np.random.choice(NAMES) + sep + np.random.choice(NAMES)
        names.append(name)
    return np.array(names)
