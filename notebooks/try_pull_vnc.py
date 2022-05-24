#%%
import pymaid

print("pymaid version:")
print(pymaid.__version__)
print()


pymaid.CatmaidInstance(
    server="https://fanc.catmaid.virtualflybrain.org/", api_token=None
)

#%%
neurons = pymaid.find_neurons(
    annotations="Paper: Phelps, Hildebrand, Graham et al. 2020"
)
connectors = pymaid.get_connectors(neurons)
pymaid.adjacency_matrix(neurons)

# pymaid.CatmaidInstance(
#     server="https://fafb.catmaid.virtualflybrain.org/", api_token=None
# )

# # n = pymaid.get_neurons(16)
# bates = pymaid.find_neurons(annotations="Paper: Bates and Schlegel et al 2020")
# len(bates)

# skids = pymaid.get_skids_by_annotation("Paper: Phelps, Hildebrand, Graham et al. 2020")
# print("# fetched neurons:")
# print(len(skids))
# print()

# connectors = pymaid.get_connectors(skids)

#%%
# %%
import pandas as pd
import numpy as np

index = pymaid.get_skids_by_annotation("Paper: Phelps, Hildebrand, Graham et al. 2020")
nodes = pd.DataFrame(index=index)
annotations = [
    "left soma",
    "right soma",
    "sensory neuron",
    "motor neuron",
    "T1 leg motor neuron",
    "T2 leg motor neuron",
    "T3 leg motor neuron",
    "wing motor neuron",
    "haltere motor neuron",
    "neck motor neuron",
]
[
    "left T1 leg nerve",
    "right T2 leg nerve",
    "central neuron",
    "chorodotonal neuron",
    "bristle",
    "T1 leg motor neuron",
    "T2 leg motor neuron",
    "T3 leg motor neuron",
]
for annotation in annotations:
    skids = pymaid.get_skids_by_annotation(annotation)
    skids = np.intersect1d(skids, index)
    print(f"Annotation: {annotation}, # found: {len(skids)}")
    nodes[annotation] = False
    nodes.loc[skids, annotation] = True

motor_nodes = nodes[nodes["motor neuron"]].copy()
motor_nodes["hemisphere"] = ""
motor_nodes.loc[motor_nodes[motor_nodes["left soma"]].index, "hemisphere"] = "L"
motor_nodes.loc[motor_nodes[motor_nodes["right soma"]].index, "hemisphere"] = "R"

# motor_nodes[motor_nodes["left soma"] | motor_nodes["right soma"]]
#%%
motor_nodes["class"] = ""

classes = [
    "T1 leg motor neuron",
    "T2 leg motor neuron",
    "T3 leg motor neuron",
    "wing motor neuron",
    "haltere motor neuron",
    "neck motor neuron",
]

for skid, row in motor_nodes.iterrows():
    annotated_as = row[classes].index[row[classes]].values
    if len(annotated_as) > 1:
        print(skid)
    else:
        annotated_as = annotated_as[0]
        annotated_as = annotated_as.replace(" motor neuron", "")
        motor_nodes.loc[skid, "class"] = annotated_as

motor_nodes.sort_values(["hemisphere", "class"], inplace=True)
# %%
adj_df = pymaid.adjacency_matrix(motor_nodes.index.values)
adj_df = pd.DataFrame(
    adj_df.values.astype(int), index=adj_df.index, columns=adj_df.columns
)
#%%
from graspologic.plot import adjplot

#%%

adjplot(adj_df.values, plot_type='scattermap')