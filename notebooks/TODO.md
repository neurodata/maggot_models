- Experiment comparing different versions of OMNI 
    - could do {iso vs aniso}
    - could do {sum of 4 colors vs. 4 colors omni separately and then concat vs. 4 colors omni together}
    - could do {svd and then cluster vertices together vs. cluster separate embeddings}
    - start by looking at naive (iso) for sum of 4-colors vs others. Run all the way 
    through the clustering 
    - depending on how that goes, maybe try the aniso version
    - future experiment: one model is we cluster on the 4 colors jointly, estimate bhat 
    separately for the same partition of the 4 color graphs. another is we estimate assignments 
    separately for each color. in terms of modeling the likelihoods of the 4 colors how do we do vs 
    number of parameters? 

- Make draft version of figure 2 (flow figure)
    - flow methods = {signal flow, graph match flow}
    - plot the 4+1 color matrices sorted by flow method
    - plot "looking down the diagonals" for each as well
    - plot the "vs random", count upper triangle figure
    - plot the rank correlation for each color
    - supplement
        - rank correlation for the two flow methods against each other

- Perturbation test 
    - randomly permute 10% of cluster labels and see if they can tell
    - run whole clustering procedure with 2 random seeds, compute ARI or something

- Make similar figure to figure 2, but using Bhat 
    - try both flow methods

- Cluster overlap in space
    - for each cluster, get the KDE of input synapse cloud and output synapse cloud

- Make nice plots of $\hat{B}$, $D^{-1}\hat{B}$, $\hat{B}D^{-1}$ 

- Follow up with Youngser stuff by Tuesday if haven't heard. Want ranks myself, figure 
  out who is not behaving well in terms of the pairs 

- plot the distribution of weights in each cluster as predicted by OMNI/GMM
    - do we need a good model for the weights or are they similarly distributed across blocks, 
    maybe with different scaling factors? 
    - maybe im asking do we have the same model family for each pair of blocks 

- Generate a list of the pairs that are not clustered into the same cluster for each level. 

