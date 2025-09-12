#!/bin/bash
# nohup sh scripts/train_render_lf.sh > logs/lf_05_gs.log 2>&1 &
# close port 3000 in the end
# netstat -anp |grep 3000
# lsof -i:3000
# kill -9 xxxx

export CUDA_VISIBLE_DEVICES=0
NAME=hs_p6_E4
EXPERIMENT=LF
DATA_DIR="/data1/fwu/3DGS/LF"
# SCENES="africa basket statue torch"
SCENES="africa"
# train process
for SCENE in $SCENES; do
  if [ -d "output/${NAME}/${EXPERIMENT}/${SCENE}" ]; then
    echo "Already trained: $SCENE"
    continue
  fi

    # python -m debugpy --listen 3000 --wait-for-client train.py \
    python -m train \
            -s "${DATA_DIR}/${SCENE}" -m "output/${NAME}/${EXPERIMENT}/${SCENE}" \
            --dataset_name $EXPERIMENT --eval \
            --resolution 2 --iterations 3000 --densify_until_iter 2000
done

