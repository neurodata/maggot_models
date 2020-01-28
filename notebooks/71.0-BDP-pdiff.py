#%%
from src.data import load_metagraph
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


mg = load_metagraph("G", "2020-01-21")

is_pdiff = np.where(mg["is_pdiff"])[0]
mg = mg.reindex(is_pdiff)
degree_df = mg.calculate_degrees()
plt.figure()
melt_degree = pd.melt(
    degree_df.reset_index(),
    id_vars=["ID"],
    value_vars=["In degree", "Out degree", "Total degree"],
    value_name="Degree",
)
sns.stripplot(y="Degree", data=melt_degree, x="variable", jitter=0.45)

plt.figure()
melt_syns = pd.melt(
    degree_df.reset_index(),
    id_vars=["ID"],
    value_vars=["In edgesum", "Out edgesum", "Total edgesum"],
    value_name="Synapses",
)
sns.stripplot(y="Synapses", data=melt_syns, x="variable", jitter=0.45)
