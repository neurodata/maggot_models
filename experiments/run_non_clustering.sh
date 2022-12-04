#!/usr/bin/env bash
BASEDIR=$(dirname "$0")
(python $BASEDIR/plot_clustered_adjacency/plot_clustered_adjacency.py) && \
(python $BASEDIR/plot_bar_dendrogram/plot_bar_dendrogram.py) && \ 
(python $BASEDIR/plot_blockmodel/plot_blockmodel.py) && \ 
(python $BASEDIR/flow_row/flow_row.py)
# adjacency matrix (2D)
# ffwd/feedback (2F)
# signal flow pairwise (S6B/2G)
# edge reciprocity (2H)
# 
# ball and stick (3D)
# paths plot (3G)
# edge overlap plot comparison to null S5D
# gm rank plot 
# flow row plot S5F
# S7B? morphology histogram
# (python $BASEDIR/nblast/nblast.py) && \
# (python $BASEDIR/plot_morphology/plot_morphology.py) && \ 