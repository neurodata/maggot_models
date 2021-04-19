#!/usr/bin/env bash
BASEDIR=$(dirname "$0")
(python $BASEDIR/flow/flow.py) && \
(python $BASEDIR/graph_match/graph_match.py) && \
(python $BASEDIR/embed/embed.py)
