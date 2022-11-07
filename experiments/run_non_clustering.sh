#!/usr/bin/env bash
BASEDIR=$(dirname "$0")
(python $BASEDIR/plot_clustered_adjacency/plot_clustered_adjacency.py) && \
(python $BASEDIR/plot_morphology/plot_morphology.py) && \ 
(python $BASEDIR/plot_bar_dendrogram/plot_bar_dendrogram.py) && \ 
(python $BASEDIR/plot_bar_dendrogram/plot_bar_dendrogram_sf.py)