#!/bin/bash
# nohup bash scripts/active_llff.sh > logs/llff_hs_active.log 2>&1 &
# close port 3000 in the end
# netstat -anp |grep 3000
# lsof -i:3000
# kill -9 xxxx
# 265639

export CUDA_VISIBLE_DEVICES=0

EXPERIMENT=llff_active
DATA_DIR="/disk1/fwu/3DGS/nerf_llff_data"

declare -A SCENE_ITERATIONS
SCENE_ITERATIONS["fern"]=9000
SCENE_ITERATIONS["flower"]=10000
SCENE_ITERATIONS["fortress"]=10500
SCENE_ITERATIONS["horns"]=12500
SCENE_ITERATIONS["leaves"]=9000
SCENE_ITERATIONS["orchids"]=9000
SCENE_ITERATIONS["room"]=10500
SCENE_ITERATIONS["trex"]=12000

# SCENES="fern flower fortress horns leaves orchids room trex"
SCENES="horns"


for SCENE in $SCENES; do
    ITERATIONS=${SCENE_ITERATIONS[$SCENE]:-7000}
#     # TEST_ITER1=$((ITERATIONS * 3 / 7))  # 约43%
#     # DENSIFY_ITER=$((ITERATIONS * 4 / 7))  # 约57%

    echo python active_train.py -s "${DATA_DIR}/${SCENE}" -m "output/${EXPERIMENT}/${SCENE}" --eval --method=HS --seed=0 --schema ${SCENE}seq1_inplace --resolution 8 --iterations $ITERATIONS --test_iterations 3000 7000 $ITERATIONS --densify_until_iter=4000 --kl_weight 1E-6 --beta_rho_scale -6.0 --sample_n 2
    
    # python -m debugpy --listen 3000 --wait-for-client active_train.py \
    python active_train.py -s "${DATA_DIR}/${SCENE}" -m "output/${EXPERIMENT}/${SCENE}" --eval --method=HS --seed=0 --schema ${SCENE}seq1_inplace --resolution 8 --iterations $ITERATIONS --test_iterations 3000 7000 $ITERATIONS --densify_until_iter=4000 --kl_weight 1E-6 --beta_rho_scale -6.0 --sample_n 2
    echo render.py -m "output/${EXPERIMENT}/${SCENE}" --skip_train
    python render.py -m "output/${EXPERIMENT}/${SCENE}" --skip_train
    echo metrics.py -m "output/${EXPERIMENT}/${SCENE}"
    python metrics.py -m "output/${EXPERIMENT}/${SCENE}"
done

# python render.py -m "output/llff_active/fern" --skip_train
# python metrics.py -m "output/llff_active/fern"