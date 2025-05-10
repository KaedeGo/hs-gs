#!/bin/bash
# nohup sh scripts/train_render_360_s.sh > train_render_360_p6.log 2>&1 &
# close port 3000 in the end
# netstat -anp |grep 3000
# lsof -i:3000
# kill -9 xxxx

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NAME=hs_p6
export CUDA_VISIBLE_DEVICES=0

EXPERIMENT=360
DATA_DIR="/home/fwu/Datasets/3DGS/360"

python train.py -s "${DATA_DIR}/bicycle" -m "output/${NAME}/360/bicycle" --eval --disable_viewer --resolution 4            
python render_uncertainty.py -m "output/${NAME}/360/bicycle" --iteration 30000 --resolution 4 --skip_train
python metrics_uncertainty.py --experiment_path "output/${NAME}/360/bicycle/test/ours_30000"

python train.py -s "${DATA_DIR}/bonsai" -m "output/${NAME}/360/bonsai" --eval --disable_viewer --resolution 2
python render_uncertainty.py -m "output/${NAME}/360/bonsai" --iteration 30000 --resolution 2 --skip_train
python metrics_uncertainty.py --experiment_path "output/${NAME}/360/bonsai/test/ours_30000"

python train.py -s "${DATA_DIR}/kitchen" -m "output/${NAME}/360/kitchen" --eval --disable_viewer --resolution 2
python render_uncertainty.py -m "output/${NAME}/360/kitchen" --iteration 30000 --resolution 2 --skip_train
python metrics_uncertainty.py --experiment_path "output/${NAME}/360/kitchen/test/ours_30000"

python train.py -s "${DATA_DIR}/counter" -m "output/${NAME}/360/counter" --eval --disable_viewer --resolution 2
python render_uncertainty.py -m "output/${NAME}/360/counter" --iteration 30000 --resolution 2 --skip_train
python metrics_uncertainty.py --experiment_path "output/${NAME}/360/counter/test/ours_30000"

python train.py -s "${DATA_DIR}/flowers" -m "output/${NAME}/360/flowers" --eval --disable_viewer --resolution 4
python render_uncertainty.py -m "output/${NAME}/360/flowers" --iteration 30000 --resolution 4 --skip_train
python metrics_uncertainty.py --experiment_path "output/${NAME}/360/flowers/test/ours_30000"

python train.py -s "${DATA_DIR}/garden" -m "output/${NAME}/360/garden" --eval --disable_viewer --resolution 4
python render_uncertainty.py -m "output/${NAME}/360/garden" --iteration 30000 --resolution 4 --skip_train
python metrics_uncertainty.py --experiment_path "output/${NAME}/360/garden/test/ours_30000"

python train.py -s "${DATA_DIR}/room" -m "output/${NAME}/360/room" --eval --disable_viewer --resolution 2
python render_uncertainty.py -m "output/${NAME}/360/room" --iteration 30000 --resolution 2 --skip_train
python metrics_uncertainty.py --experiment_path "output/${NAME}/360/room/test/ours_30000"

python train.py -s "${DATA_DIR}/stump" -m "output/${NAME}/360/stump" --eval --disable_viewer --resolution 4
python render_uncertainty.py -m "output/${NAME}/360/stump" --iteration 30000 --resolution 4 --skip_train
python metrics_uncertainty.py --experiment_path "output/${NAME}/360/stump/test/ours_30000"

python train.py -s "${DATA_DIR}/treehill" -m "output/${NAME}/360/treehill" --eval --disable_viewer --resolution 4
python render_uncertainty.py -m "output/${NAME}/360/treehill" --iteration 30000 --resolution 4 --skip_train
python metrics_uncertainty.py --experiment_path "output/${NAME}/360/treehill/test/ours_30000"


