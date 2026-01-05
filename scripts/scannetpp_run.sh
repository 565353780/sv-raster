# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

DATA_ROOT=data/scannetpp_nvs

lanuch_exp() {
    local scene_name="$1"
    shift
    local output_dir="$1"
    shift
    local exp_args="$*"

    python train.py --source_path $DATA_ROOT/$scene_name --model_path $output_dir/$scene_name $exp_args
    python render.py $output_dir/$scene_name --skip_train --eval_fps
    python render.py $output_dir/$scene_name --skip_train
    python eval.py $output_dir/$scene_name
    python render_fly_through.py $output_dir/$scene_name
    rm -r $output_dir/$scene_name/checkpoints/
}

ulimit -n 4096  # Increase maximum number of files the script can read

for scene in 39f36da05b 5a269ba6fe dc263dfbf0 08bbbdcc3d
do
    echo "============ start " $scene " ============"
    if [ ! -f $1/$scene/results.json ]; then
        # We use the source image resolution and prevent automatic downsampling.
        lanuch_exp $scene $1 --res_downscale 1.0 --cfg_files cfg/scannetpp.yaml "${@:2}"
    fi
    echo "============ end " $scene " ============"
done
