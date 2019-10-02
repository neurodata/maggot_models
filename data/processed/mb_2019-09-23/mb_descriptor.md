# _Drosophila_ Larva Mushroom Body Connectome

This folder contains ``.graphml`` files for the _Drosophila_ larva connectome from
Eichler et al. 2017, Nature. 

## Multi-channel network
The connectomes have been updated by the author's since the original paper release. In 
addition, they have separated synapses into categories based on the part of the neuron
which is pre- and post-synaptic. Synapses can be: 
- axo-axonic (aa)
- axo-dendritic (ad) (Cajalian) 
- dendro-axonic (da)
- dendro-dendritic (dd)

We will refer to the union of these 4 types as the "full" graph.

## Normalization
Normalization, as suggested by the authors who collected the data, is as follows: raw 
edge weights are divided by the sum of input to that compartment for that neuron. That
is, the number of synapses from Cell A's axon or a dendrite onto Cell B's axon would be
divided by the total number of synapses onto Cell B's axon. The is is the resulting edge
weight in the normalized graphs. For the full graph (union of the 4 types of synapses),
the normalization is done by the sum of dendrite and axon input for that cell. 

## File Naming 
- G: the raw-weight graph for the full graph
- Gaa: the raw-weight graph for axo-axonic synapses.
- Gad: the raw-weight graph for axo-dendritic synapses. 
- Gda: the raw-weight graph for dendro-dendritic synapses. 
- Gdd: the raw-weight graph for dendro-axonic synapses. 
- Gn: the normalized-weight graph for the full graph
- Gaan: the normalized-weight graph for the axo-axonic synapses.
- Gadn: the normalized-weight for axo-dendritic synapses. 
- Gdan: the normalized-weight for dendro-dendritic synapses. 
- Gddn: the normalized-weight for dendro-axonic synapses.

## Metadata
The ``.graphml`` files contain attributes for each node. The attribute types are as
follows: 
- Hemisphere: which side of the brain the neuron is on (left/right/center)
- Class: morphologically/anatomically defined cell types. Possible values are:
    - 'PN'
        - 'Gustatory PN'
        - 'ORN mPN'
        - 'ORN uPN'
        - 'tPN'
        - 'vPN'
        - 'Unknown PN'
    - 'MBIN'
    - 'APL'
    - 'MBON'
    - 'KC'
        - 'KC 1 claw'
        - 'KC 2 claw'
        - 'KC 3 claw'
        - 'KC 4 claw'
        - 'KC 5 claw'
        - 'KC 6 claw'
        - 'KC young'
 
- Pair: the name of a left/right pair that the neuron belongs to, not all neurons will 
have a pair (those that don't should be KCs)

Nodes are indexed by integer IDs corresponding to their skeleton in the original data