#!/bin/bash
# bash scripts/eval.sh

DIR_OUT="/home/fwu/Documents/myProjects/hs-gs/eval"

EXPERIMENT="db"
# EXPERIMENT="360 db tandt nerf_synthetic"
SCENES_360="bicycle bonsai counter flowers garden kitchen room stump treehill"
SCENES_db="drjohnson"
# SCENES_db="drjohnson playroom"
SCENES_tandt="train truck"
SCENES_nerf_synthetic="chair drums ficus hotdog lego materials mic ship"


for exp in $EXPERIMENT; do
    for scene in $(eval echo \$SCENES_$exp); do
        python metrics.py -m $DIR_OUT/$exp/$scene
    done
done