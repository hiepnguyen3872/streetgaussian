#!/bin/bash
# scenes=("090" "105" "108" "134" "150" "181" "026" "006") 
# scenes=("090" "150") 
# scenes=("150" "105" "108" "134" "181" "026" "006") 
# scenes=("105" "026" "150")
# scenes=("026")
# scenes=("090")
# scenes=("150" "105") 
# scenes=("105") 
scenes=("150") 
export QT_SELECTION=/usr/bin/qmake
export QT_QPA_PLATFORM=offscreen
unset QT_QPA_PLATFORM
for scene in "${scenes[@]}"; do
    # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python train.py --config configs/experiments_waymo/street_gaussian_wo_3DOD/waymo_val_$scene.yaml
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python train.py --config configs/experiments_waymo/semantic_and_pseudo_depth/waymo_val_$scene.yaml
    # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train_pvg.py --config configs/experiments_waymo/pvg/waymo_val_$scene.yaml
    # python render.py --config configs/experiments_waymo/pvg/waymo_val_$scene.yaml mode evaluate
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python render.py --config configs/experiments_waymo/semantic_and_pseudo_depth/waymo_val_$scene.yaml mode evaluate
    # python render.py --config configs/experiments_waymo/waymo_val_deformable_$scene.yaml mode trajectory
    # python make_ply.py --config configs/experiments_waymo/waymo_val_deformable_$scene.yaml viewer.frame_id 0 mode evaluate
done
