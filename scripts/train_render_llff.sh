#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

EXPERIMENT=llff
DATA_DIR="/path/to/nerf_llff_data"
SCENES="fern flower fortress horns leaves orchids room trex"

for SCENE in $SCENES; do
  if [ -d "output/${EXPERIMENT}/${SCENE}" ]; then
    echo "Already trained: $SCENE"
    continue
  fi
  python train.py \
          -s "${DATA_DIR}/${SCENE}" -m "output/${EXPERIMENT}/${SCENE}" \
          --dataset_name $EXPERIMENT --eval \
          --resolution 8 --iterations 7000 --densify_until_iter 4000 \
          --kl_weight 1E-3 --beta_rho_scale -5.0 --sample_n 2
done
