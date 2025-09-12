#!/bin/bash
# nohup sh scripts/train_render_llff.sh > logs/llff_05_gs.log 2>&1 &
# close port 3000 in the end
# netstat -anp |grep 3000
# lsof -i:3000
# kill -9 xxxx

export CUDA_VISIBLE_DEVICES=7
NAME=hs_p6_E4
EXPERIMENT=llff
DATA_DIR="/disk1/fwu/3DGS/nerf_llff_data"
SCENES="fern flower fortress horns leaves orchids room trex"
# SCENES="fern"
# train process
for SCENE in $SCENES; do
  if [ -d "output/${NAME}/${EXPERIMENT}/${SCENE}" ]; then
    echo "Already trained: $SCENE"
    continue
  fi

  python train.py \
          -s "${DATA_DIR}/${SCENE}" -m "output/${NAME}/${EXPERIMENT}/${SCENE}" \
          --dataset_name $EXPERIMENT --eval \
          --resolution 8 --iterations 7000 --densify_until_iter 4000
done
