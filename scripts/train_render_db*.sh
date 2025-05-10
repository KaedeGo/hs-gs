#!/bin/bash
# nohup sh scripts/train_render_db*.sh > train_render_db_*.log 2>&1 &
# close port 3000 in the end
# netstat -anp |grep 3000
# lsof -i:3000
# kill -9 xxxx

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=1
NAME=hs_*

EXPERIMENT=db
DATA_DIR="/home/fwu/Datasets/3DGS/db"
# SCENES="playroom"
SCENES="drjohnson playroom"
# RUN_CMD="python3 -m debugpy --listen 3000 --wait-for-client"
# train process
for SCENE in $SCENES; do
  if [ -d "output/${NAME}/${EXPERIMENT}/${SCENE}" ]; then
    echo "Already trained: $SCENE"
    continue
  fi

  python train.py -s "${DATA_DIR}/${SCENE}" -m "output/${NAME}/${EXPERIMENT}/${SCENE}" --eval --disable_viewer --resolution 4 
done

# render process
for SCENE in $SCENES; do
  if [ ! -d "output/hs_*/${EXPERIMENT}/${SCENE}/test/ours_30000" ]; then
    echo "Rendering: $SCENE at iteration 30000"
    python render_uncertainty.py -m "output/${NAME}/${EXPERIMENT}/${SCENE}" --iteration 30000 --resolution 4
    python metrics_uncertainty.py --experiment_path "output/${NAME}/${EXPERIMENT}/${SCENE}/test/ours_30000"
  else
    echo "Already rendered: $SCENE (30000)"
  fi
done


