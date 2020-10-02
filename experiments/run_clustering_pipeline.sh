#!/usr/bin/env bash
BASEDIR=$(dirname "$0")
# echo "$BASEDIR"
(python $BASEDIR/graph_match/graph_match_hemispheres.py) && \
(python $BASEDIR/matched_subgraph_omni_cluster/matched_subgraph_omni_cluster.py) && \
(python $BASEDIR/evaluate_clustering/evaluate_clustering.py)
