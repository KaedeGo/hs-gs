#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

EXPERIMENT=LF
DATA_DIR="/path/to/LF"
SCENES="africa basket statue torch"

# train process
for SCENE in $SCENES; do
  if [ -d "output/${EXPERIMENT}/${SCENE}" ]; then
    echo "Already trained: $SCENE"
    continue
  fi

    python train.py \
            -s "${DATA_DIR}/${SCENE}" -m "output/${EXPERIMENT}/${SCENE}" \
            --dataset_name $EXPERIMENT --eval \
            --resolution 2 --iterations 3000 --densify_until_iter 2000 \
            --kl_weight 1E-3 --beta_rho_scale -5.0 --sample_n 64
done



