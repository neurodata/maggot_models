#%%

import numpy as np

a = np.array(
    [[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1]]
)
a

#%%

for level in range(1, 3):
    uni_labels, inv = np.unique(a[:, :level], axis=0, return_inverse=True)
    inv
    a[:, 1] = inv

#%%

for nodes in LevelOrderGroupIter(root):
    for i, node in enumerate(nodes):
        node._label = i
