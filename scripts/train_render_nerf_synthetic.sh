#!/bin/bash
# sh scripts/train_render_nerf_synthetic.sh
# close port 3000 in the end
# netstat -anp |grep 3000
# lsof -i:3000
# kill -9 xxxx

export CUDA_VISIBLE_DEVICES=2

EXPERIMENT=nerf_synthetic
DATA_DIR="/home/fwu/Datasets/3DGS/nerf_synthetic"
SCENES="chair drums ficus hotdog lego materials mic ship"

# train process
for SCENE in $SCENES; do
  if [ -d "output/${EXPERIMENT}/${SCENE}" ]; then
    echo "Already trained: $SCENE"
    continue
  fi

  python train.py -s "${DATA_DIR}/${SCENE}" -m "output/${EXPERIMENT}/${SCENE}" --eval --disable_viewer
done

# render process
for SCENE in $SCENES; do
  if [ ! -d "output/${EXPERIMENT}/${SCENE}/test/ours_30000" ]; then
    echo "Rendering: $SCENE at iteration 30000"
    python render.py -m "output/${EXPERIMENT}/${SCENE}" --iteration 30000
  else
    echo "Already rendered: $SCENE (30000)"
  fi

  if [ ! -d "output/${EXPERIMENT}/${SCENE}/test/ours_7000" ]; then
    echo "Rendering: $SCENE at iteration 7000"
    python render.py -m "output/${EXPERIMENT}/${SCENE}" --iteration 7000
  else
    echo "Already rendered: $SCENE (7000)"
  fi

done


