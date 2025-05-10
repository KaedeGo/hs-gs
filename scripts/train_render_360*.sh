#!/bin/bash
# nohup sh scripts/train_render_360*.sh > train_render_360.log 2>&1 &
# close port 3000 in the end
# netstat -anp |grep 3000
# lsof -i:3000
# kill -9 xxxx

export CUDA_VISIBLE_DEVICES=1

EXPERIMENT=360
DATA_DIR="/home/fwu/Datasets/3DGS/360"
NAME=hs_*

# OUTDOOR_SCENES="bicycle garden stump"
OUTDOOR_SCENES="bicycle flowers garden stump treehill"

# INDOOR_SCENES="kitchen room counter"
INDOOR_SCENES="kitchen room bonsai counter"

# train process
for SCENE in $INDOOR_SCENES; do
  if [ -d "output/${NAME}/${EXPERIMENT}/${SCENE}" ]; then
    echo "Already trained: $SCENE"
    continue
  fi

  python train.py -s "${DATA_DIR}/${SCENE}" -m "output/${NAME}/${EXPERIMENT}/${SCENE}" --eval --disable_viewer --resolution 8
done

for SCENE in $OUTDOOR_SCENES; do
  if [ -d "output/${NAME}/${EXPERIMENT}/${SCENE}" ]; then
    echo "Already trained: $SCENE"
    continue
  fi

  python train.py -s "${DATA_DIR}/${SCENE}" -m "output/${NAME}/${EXPERIMENT}/${SCENE}" --eval --disable_viewer --resolution 8
done

# render process
for SCENE in $INDOOR_SCENES; do
  if [ ! -d "output/${NAME}/${EXPERIMENT}/${SCENE}/test/ours_30000" ]; then
    echo "Rendering: $SCENE at iteration 30000"
    python render_uncertainty.py -m "output/${NAME}/${EXPERIMENT}/${SCENE}" --iteration 30000 --resolution 8 --skip_train
    python metrics_uncertainty.py --experiment_path "output/${NAME}/${EXPERIMENT}/${SCENE}/test/ours_30000"
  else
    echo "Already rendered: $SCENE (30000)"
  fi

  # if [ ! -d "output/hs/${EXPERIMENT}/${SCENE}/test/ours_7000" ]; then
  #   echo "Rendering: $SCENE at iteration 7000"
  #   python render_uncertainty.py -m "output/hs/${EXPERIMENT}/${SCENE}" --iteration 7000 --resolution 2 --skip_train
  # else
  #   echo "Already rendered: $SCENE (7000)"
  # fi

done

for SCENE in $OUTDOOR_SCENES; do
  if [ ! -d "output/${NAME}/${EXPERIMENT}/${SCENE}/test/ours_30000" ]; then
    echo "Rendering: $SCENE at iteration 30000"
    python render_uncertainty.py -m "output/${NAME}/${EXPERIMENT}/${SCENE}" --iteration 30000 --resolution 8 --skip_train
    python metrics_uncertainty.py --experiment_path "output/${NAME}/${EXPERIMENT}/${SCENE}/test/ours_30000"
  else
    echo "Already rendered: $SCENE (30000)"
  fi

  # if [ ! -d "output/hs/${EXPERIMENT}/${SCENE}/test/ours_7000" ]; then
  #   echo "Rendering: $SCENE at iteration 7000"
  #   python render_uncertainty.py -m "output/hs/${EXPERIMENT}/${SCENE}" --iteration 7000 --resolution 4 --skip_train
  # else
  #   echo "Already rendered: $SCENE (7000)"
  # fi

done


