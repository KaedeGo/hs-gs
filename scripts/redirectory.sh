#!/bin/bash
# bash scripts/redirectory.sh

DIR_INGS="/home/fwu/Documents/myProjects/hs-gs/output/ingp"
DIR_3DGS="/home/fwu/Documents/myProjects/hs-gs/output/3dgs"
DIR_SVOX2="/home/fwu/Documents/myProjects/hs-gs/output/svox2"
DIR_HS="/home/fwu/Documents/myProjects/hs-gs/output/hs"

DIR_OUT="/home/fwu/Documents/myProjects/hs-gs/eval"
baselines="hs_7k hs_30k"
# baselines="3dgs_7k 3dgs_30k ingp_base ingp_big svox2 hs_7k hs_30k"

# EXPERIMENT="tandt"
EXPERIMENT="360 db nerf_synthetic"
SCENES_360="bicycle bonsai counter flowers garden kitchen room stump treehill"
SCENES_db="drjohnson playroom"
SCENES_tandt="train truck"
SCENES_nerf_synthetic="chair drums ficus hotdog lego materials mic ship"

for exp in $EXPERIMENT; do
    for scene in $(eval echo \$SCENES_$exp); do
        if [ -d "$DIR_OUT/$exp/$scene/test" ]; then
            echo "Directory $DIR_OUT/$exp/$scene/test already exists"
        else
            mkdir -p "$DIR_OUT/$exp/$scene/test"
        fi
        
        for baseline in $baselines; do
            if [ -d "$DIR_OUT/$exp/$scene/test/$baseline" ]; then
                echo "Directory $DIR_OUT/$exp/$scene/test/$baseline already exists"
                continue
            else
                mkdir -p "$DIR_OUT/$exp/$scene/test/$baseline"
                echo "Created directory $DIR_OUT/$exp/$scene/test/$baseline"
                
                if [ "$baseline" == "3dgs_7k" ]; then
                    cp -r "$DIR_3DGS/$exp/$scene/test/ours_7000/gt" "$DIR_OUT/$exp/$scene/test/$baseline"
                    cp -r "$DIR_3DGS/$exp/$scene/test/ours_7000/renders" "$DIR_OUT/$exp/$scene/test/$baseline"

                elif [ "$baseline" == "3dgs_30k" ]; then
                    cp -r "$DIR_3DGS/$exp/$scene/test/ours_30000/gt" "$DIR_OUT/$exp/$scene/test/$baseline"
                    cp -r "$DIR_3DGS/$exp/$scene/test/ours_30000/renders" "$DIR_OUT/$exp/$scene/test/$baseline"

                elif [ "$baseline" == "ingp_base" ]; then
                    cp -r "$DIR_INGS/$exp/$scene/base/gt" "$DIR_OUT/$exp/$scene/test/$baseline"
                    cp -r "$DIR_INGS/$exp/$scene/base/renders" "$DIR_OUT/$exp/$scene/test/$baseline"
                
                elif [ "$baseline" == "ingp_big" ]; then
                    cp -r "$DIR_INGS/$exp/$scene/big/gt" "$DIR_OUT/$exp/$scene/test/$baseline"
                    cp -r "$DIR_INGS/$exp/$scene/big/renders" "$DIR_OUT/$exp/$scene/test/$baseline"

                elif [ "$baseline" == "svox2" ]; then
                    cp -r "$DIR_SVOX2/$exp/$scene/test_renders/gt" "$DIR_OUT/$exp/$scene/test/$baseline"
                    cp -r "$DIR_SVOX2/$exp/$scene/test_renders/renders" "$DIR_OUT/$exp/$scene/test/$baseline"
                
                elif [ "$baseline" == "hs_7k" ]; then
                    cp -r "$DIR_HS/$exp/$scene/test/ours_7000/gt" "$DIR_OUT/$exp/$scene/test/$baseline"
                    cp -r "$DIR_HS/$exp/$scene/test/ours_7000/renders" "$DIR_OUT/$exp/$scene/test/$baseline"
                
                elif [ "$baseline" == "hs_30k" ]; then
                    cp -r "$DIR_HS/$exp/$scene/test/ours_30000/gt" "$DIR_OUT/$exp/$scene/test/$baseline"
                    cp -r "$DIR_HS/$exp/$scene/test/ours_30000/renders" "$DIR_OUT/$exp/$scene/test/$baseline"
                
                else
                    echo "Unknown baseline $baseline"
                fi
            fi
        done
    done
done