#!/bin/bash
# scenes=("090" "105" "108" "134" "150" "181") 
scenes=("006" "026") 

for scene in "${scenes[@]}"; do
    echo "preprocess scene $scene"
    CUDA_VISIBLE_DEVICES=1 python script/waymo/generate_sky_mask.py --datadir "/media/ml4u/ExtremeSSD/datasets/waymo/processed/$scene" --sam_checkpoint "/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/sam_vit_h_4b8939.pth"
    CUDA_VISIBLE_DEVICES=1 python script/waymo/generate_lidar_depth.py --datadir "/media/ml4u/ExtremeSSD/datasets/waymo/processed/$scene"
    CUDA_VISIBLE_DEVICES=1 python script/waymo/generate_mono_depth.py --input_dir "/media/ml4u/ExtremeSSD/datasets/waymo/processed/$scene/images" --output_dir "/media/ml4u/ExtremeSSD/datasets/waymo/processed/$scene"
done
