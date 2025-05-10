#!/bin/bash
# nohup sh scripts/train_render_nerf_synthetic*.sh > train_render_nerf_synthetic_*.log 2>&1 &
# close port 3000 in the end
# netstat -anp |grep 3000
# lsof -i:3000
# kill -9 xxxx

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2
NAME=hs_*
EXPERIMENT=nerf_synthetic
DATA_DIR="/home/fwu/Datasets/3DGS/nerf_synthetic"
SCENES="chair drums ficus hotdog lego materials mic ship"

# train process
for SCENE in $SCENES; do
  if [ -d "output/${NAME}/${EXPERIMENT}/${SCENE}" ]; then
    echo "Already trained: $SCENE"
    continue
  fi

  python train.py -s "${DATA_DIR}/${SCENE}" -m "output/${NAME}/${EXPERIMENT}/${SCENE}" --eval --disable_viewer
done

# render process
for SCENE in $SCENES; do
  if [ ! -d "output/${NAME}/${EXPERIMENT}/${SCENE}/test/ours_30000" ]; then
    echo "Rendering: $SCENE at iteration 30000"
    python render_uncertainty.py -m "output/${NAME}/${EXPERIMENT}/${SCENE}" --iteration 30000
    python metrics_uncertainty.py --experiment_path "output/${NAME}/${EXPERIMENT}/${SCENE}/test/ours_30000"
  else
    echo "Already rendered: $SCENE (30000)"
  fi

  # if [ ! -d "output/hs/${EXPERIMENT}/${SCENE}/test/ours_7000" ]; then
  #   echo "Rendering: $SCENE at iteration 7000"
  #   python render_uncertainty.py -m "output/hs/${EXPERIMENT}/${SCENE}" --iteration 7000
  # else
  #   echo "Already rendered: $SCENE (7000)"
  # fi

done

