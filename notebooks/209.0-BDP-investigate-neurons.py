#%%
from src.data import load_metagraph

mg = load_metagraph("G")

meta = mg.meta

#%%
mbin = meta[meta["class1"] == "MBIN"]

mbin

mbin.loc[[4381377, 10673895]]["name"]

#%%
from src.visualization import simple_plot_neurons
from src.pymaid import start_instance
from src.visualization import CLASS_COLOR_DICT as palette
import matplotlib.pyplot as plt

start_instance()
fig = plt.figure(figsize=(6, 6))
gs = plt.GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[(0, 0)], projection="3d")
neuron_palette = dict(zip(meta.index.values, list(meta["merge_class"].map(palette))))
simple_plot_neurons([4381377, 10673895], palette=neuron_palette, ax=ax)

fig = plt.figure(figsize=(6, 6))
gs = plt.GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[(0, 0)], projection="3d")
simple_plot_neurons(list(mbin.index.values), palette=neuron_palette, ax=ax)
