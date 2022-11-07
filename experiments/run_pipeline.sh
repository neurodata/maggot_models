#!/usr/bin/env bash
BASEDIR=$(dirname "$0")
# (python $BASEDIR/define_nodes/define_nodes.py) && \
# (python $BASEDIR/flow/flow.py) && \
# (python $BASEDIR/walk_sort/generate_walks.py) && \
# (python $BASEDIR/walk_sort/walk_sort.py) && \
# (python $BASEDIR/graph_match/graph_match_grouped.py) && \
# (python $BASEDIR/revamp_embed/revamp_embed.py) && \
# (python $BASEDIR/gaussian_cluster/gaussian_cluster.py)
(python $BASEDIR/plot_clustered_adjacency/plot_clustered_adjacency.py) && \
(python $BASEDIR/plot_morphology/plot_morphology.py) && \ 
(python $BASEDIR/plot_bar_dendrogram/plot_bar_dendrogram.py) && \ 
(python $BASEDIR/plot_bar_dendrogram/plot_bar_dendrogram_sf.py)