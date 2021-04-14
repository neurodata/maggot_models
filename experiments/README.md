# Experiment scripts
These are somewhat-solidified processes we're always going to run on the updated data.
Many of these scripts depend on previous ones. 

## Descriptions
- `flow`
    - Anything regarding ranking the nodes of the network in order from sensory to motor
    - Writes outputs as columns in the node metadata
- `graph_match`
    - Used to predict neuron pairings for the unpaired neurons
    - Plots the results of the matching
    - Writes outputs as columns in the node metadata
- `nblast`
    - Runs NBLAST (currently only within hemispheres)
    - Saves outputs as csv similarity/score matrices
- `embed`
    - Performs various embeddings on the data
    - Saves outputs as csv matrices
    - Generates plots
    - Requirements: `graph_match`
- `cluster`
    - Performs a clustering on the data
    - Writes outputs as columns in the node metadata
    - Requirements: `embed`
- `cluster_metrics`
    - Calculates various properties of the clustering
    - Generates plots
    - TODO saves models fit on the data
    - Requirements: `cluster`, `flow`
- `cluster_morphology`
    - Displays the morphology of each cluster
    - Quantifies the morphological similarity within each cluster
    - Generates plots
    - Requirements: `cluster`, `nblast`, `flow`
- `subcluster_morphology`
    - Further clusters each of the connectivity based clusters based on morphology
