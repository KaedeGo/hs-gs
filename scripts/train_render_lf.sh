#!/bin/bash
# nohup sh scripts/train_render_lf.sh > logs/lf_hs.log 2>&1 &
# close port 3000 in the end
# netstat -anp |grep 3000
# lsof -i:3000
# kill -9 xxxx

export CUDA_VISIBLE_DEVICES=0

EXPERIMENT=LF
DATA_DIR="/disk1/fwu/3DGS/LF"
SCENES="africa basket statue torch"
# SCENES="africa"

kl_weight="1E-1 1E-2 1E-3 1E-4 1E-5 1E-6 1E-7 1E-8"
beta_rho_scale="-6.0 -5.0 -4.0 -3.0 -2.0 -1.0 -0.5"
sample_n="2 4 8 16 32 64"

# kl_weight="1E-4"
# beta_rho_scale="-6.0"
# sample_n="8 16"

for kl_weight in $kl_weight; do
  for beta_rho_scale in $beta_rho_scale; do
    for sample_n in $sample_n; do
      NAME=hs_p${beta_rho_scale}_E${kl_weight}_n_${sample_n}
      # train process
      for SCENE in $SCENES; do
        if [ -d "output/${NAME}/${EXPERIMENT}/${SCENE}" ]; then
          echo "Already trained: $SCENE"
          continue
        fi
        echo "python -m train \
                  -s "${DATA_DIR}/${SCENE}" -m "output/${NAME}/${EXPERIMENT}/${SCENE}" \
                  --dataset_name $EXPERIMENT --eval \
                  --resolution 2 --iterations 3000 --densify_until_iter 2000 \
                  --kl_weight ${kl_weight} --beta_rho_scale ${beta_rho_scale} --sample_n ${sample_n}"
        
        python -m train \
                -s "${DATA_DIR}/${SCENE}" -m "output/${NAME}/${EXPERIMENT}/${SCENE}" \
                --dataset_name $EXPERIMENT --eval \
                --resolution 2 --iterations 3000 --densify_until_iter 2000 \
                --kl_weight ${kl_weight} --beta_rho_scale ${beta_rho_scale} --sample_n ${sample_n}
      done
      rm -rf "output/${NAME}/${EXPERIMENT}"
    done
  done
done


# NAME=hs_p5_E5_n_8
# EXPERIMENT=LF
# DATA_DIR="/disk1/fwu/3DGS/LF"

# # train process
# for SCENE in $SCENES; do
#   if [ -d "output/${NAME}/${EXPERIMENT}/${SCENE}" ]; then
#     echo "Already trained: $SCENE"
#     continue
#   fi

#     python -m train \
#     # python -m debugpy --listen 3000 --wait-for-client train.py \
#             -s "${DATA_DIR}/${SCENE}" -m "output/${NAME}/${EXPERIMENT}/${SCENE}" \
#             --dataset_name $EXPERIMENT --eval \
#             --resolution 2 --iterations 3000 --densify_until_iter 2000 \
#             --kl_weight 1E-5 --beta_rho_scale -5.0 --sample_n 8
# done



