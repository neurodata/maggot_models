- Experiment comparing different versions of OMNI 
    - could do {iso vs aniso}
    - could do {sum of 4 colors vs. 4 colors omni separately and then concat vs. 4 colors omni together}
    - could do {svd and then cluster vertices together vs. cluster separate embeddings}
    - depending on how that goes, maybe try the aniso version 
    - future experiment: one model is we cluster on the 4 colors jointly, estimate bhat 
    separately for the same partition of the 4 color graphs. another is we estimate assignments 
    separately for each color. in terms of modeling the likelihoods of the 4 colors how do we do vs 
    number of parameters? 


- Perturbation test 
    - run whole clustering procedure with 2 random seeds, compute ARI or something


- Make nice plots of $\hat{B}$, $D^{-1}\hat{B}$, $\hat{B}D^{-1}$ 

- Follow up with Youngser stuff by Tuesday if haven't heard. Want ranks myself, figure 
  out who is not behaving well in terms of the pairs [DONE]

- plot the distribution of weights in each cluster as predicted by OMNI/GMM
    - do we need a good model for the weights or are they similarly distributed across blocks, 
    maybe with different scaling factorUsing multiple-representation learning techniques to cluster on this graph (SPLITTER, BERT)? 
    - maybe im asking do we have the same model family for each pair of blocks 

- Generate a list of the pairs that are not clustered into the same cluster for each level. 

- could do a comparison of single-level or multilevel likelihoods as a function of embedding
  dimension. this might help pick an embedding dimension or a method of doing the embedding


--- 
- really need to do some kinda signal flow or graph match flow, possibly on samples from 
  the model I generate. 

- could try dcorr for identifying PNs