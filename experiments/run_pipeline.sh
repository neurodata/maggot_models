#!/usr/bin/env bash
BASEDIR=$(dirname "$0")
# (python $BASEDIR/flow/flow.py) && \
# (python $BASEDIR/graph_match/graph_match_grouped.py) && \
(python $BASEDIR/revamp_embed/revamp_embed.py) && \
(python $BASEDIR/gaussian_cluster/gaussian_cluster.py)