#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

EXPERIMENT=llff_active
DATA_DIR="/path/to/nerf_llff_data"

declare -A SCENE_ITERATIONS
SCENE_ITERATIONS["fern"]=9000
SCENE_ITERATIONS["flower"]=10000
SCENE_ITERATIONS["fortress"]=10500
SCENE_ITERATIONS["horns"]=12500
SCENE_ITERATIONS["leaves"]=9000
SCENE_ITERATIONS["orchids"]=9000
SCENE_ITERATIONS["room"]=10500
SCENE_ITERATIONS["trex"]=12000

SCENES="fern flower fortress horns leaves orchids room trex"

for SCENE in $SCENES; do
    ITERATIONS=${SCENE_ITERATIONS[$SCENE]:-7000}

    echo python active_train.py -s "${DATA_DIR}/${SCENE}" -m "output/${EXPERIMENT}/${SCENE}" --eval --method=HS --seed=0 --schema ${SCENE}seq1_inplace --resolution 8 --iterations $ITERATIONS --test_iterations 3000 7000 $ITERATIONS --densify_until_iter=4000 --kl_weight 1E-3 --beta_rho_scale -5.0 --sample_n 2
    
    python active_train.py -s "${DATA_DIR}/${SCENE}" -m "output/${EXPERIMENT}/${SCENE}" --eval --method=HS --seed=0 --schema ${SCENE}seq1_inplace --resolution 8 --iterations $ITERATIONS --test_iterations 3000 7000 $ITERATIONS --densify_until_iter=4000 --kl_weight 1E-3 --beta_rho_scale -5.0 --sample_n 2
    echo render.py -m "output/${EXPERIMENT}/${SCENE}" --skip_train
    python render.py -m "output/${EXPERIMENT}/${SCENE}" --skip_train
    echo metrics.py -m "output/${EXPERIMENT}/${SCENE}"
    python metrics.py -m "output/${EXPERIMENT}/${SCENE}"
done