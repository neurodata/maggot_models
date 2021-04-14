# Experiment scripts
These are somewhat-solidified processes we're always going to run on the updated data.
Many of these scripts depend on previous ones. 

## Prerequisites 
- `process_maggot_brain_connectome_<date>`
- TODO: `manage_data`
    - Option to push to a server
    - Option to upload somewhere
    - Option to change the default data version
        - TODO: make the version its own text file

## Analysis scripts
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
    - TODO: saves models fit on the data
    - Requirements: `cluster`, `flow`
- TODO: `cluster_morphology`
    - Displays the morphology of each cluster
    - Quantifies the morphological similarity within each cluster
    - Generates plots
    - Requirements: `cluster`, `nblast`, `flow`
- TODO: `subcluster_morphology`
    - Further clusters each of the connectivity based clusters based on morphology
    - Requrements: `cluster`, `nblast`
- TODO: `layout`
    - Creates a 2D graph layout
    - Requirements: `embed`

## Pipelines
Shell scripts for chaining together some of the above operations
TODO use luigi or some actutal framework for this? Or just throw it together with scripts?
