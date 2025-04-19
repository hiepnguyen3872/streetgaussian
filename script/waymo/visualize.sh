#!/bin/bash
# scenes=("006"  "026" "105" "108" "134" "150" "181") 
scenes=("090") 
# scenes=("026" "105" "150") 

for scene in "${scenes[@]}"; do
    # python render.py --config configs/experiments_waymo/semantic/waymo_val_deformable_semantic_$scene.yaml mode evaluate
    python render.py --config /home/ml4u/BKTeam/TheHiep/street_gaussians/output/waymo_exp/waymo_val_$scene/configs/config_000000.yaml mode evaluate
    # python visualize.py --video_name visualize_$scene --image_path "/home/ml4u/BKTeam/TheHiep/street_gaussians/output/waymo_exp/waymo_val_$scene"
done