# Cell type classes

Sensory
---
- `sens`: anything labeled as a sensory neuron
   - Usually just get the axon entering the volume in this dataset
   - There are several nerve bundles coming in
   - Subclasses:
      - `ORN`: odorant receptor
         - Paper: Berck et al.
         - These are in the antennal nerve also
      - `AN`: antennal nerve (anatomical)
         - Paper: https://elifesciences.org/articles/40247
         - In the antennal nerve but not odor receptors
         - Pharyngeal/internal organ info
      - `MN`: maxillary nerve (anatomical)
         - Paper: https://elifesciences.org/articles/40247
         - Gustatory/somatosensory
      - `PaN`: prothoracic accessory nerve (anatomical)
         - Paper: https://elifesciences.org/articles/40247
         - Gustatory/somatosensory
      - `Photo`: photoreceptor
         - Paper: https://elifesciences.org/articles/28387
      - `Temp`: temperature sensing
         - Paper: no paper
         - Come in as part of AN (I think)

Output 
--- 
- All of these will have the `O_` tag at the beginning
- `dVNC`: descending to ventral nerve cord
- `dSEZ`: descending to sub-esophageal zone
- `RG`: ring gland
   - Hormone outputting organ, located near top of the brain
   - Subclasses: 
      - `IPC`: insulin producing cells
         - Paper: https://elifesciences.org/articles/16799
      - `ITP`
      - `CA-LP`
      - `CRZ` 
      - `DMS`
- Motor neurons in SEZ (don't have these yet)

Mushroom body
--- 
- Clearly anatomically defined region (for the most part)
- Paper: [Eichler et al. 2017](https://www.nature.com/articles/nature23455)
- `KC`: Kenyon cells
   - Subclasses:
      - `-{claw number}`
- `MBIN`: Mushroom body input neuron
- `MBON`: Mushroom body output neuron

Antennal lobe
--- 
- Anatomical region
- Paper: [Berck et al. 2016](https://elifesciences.org/articles/14859)
- `bLN`
   - `Trio`
   - `Duet`
- `cLN`
   - `choosy`
   - `picky`
- `keystone`

Lateral horn and convergence neurons
---
- No paper
- Defined based on normalized input thresholding

Feedback neurons
---
- Claire's paper