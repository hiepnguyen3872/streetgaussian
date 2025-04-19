################################################################################

import argparse
import os
import re
import cv2
import numpy as np
import time
from skimage.util import random_noise
import glob2

if __name__ == "__main__": 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # out = cv2.VideoWriter('viz_pseudo_depth_metric3d.mp4',fourcc, 15.0, (1024*2, 512))
    # # scenes=["006",  "026", "090", "105", "108" ,"134", "150" ,"181"]
    # # scenes=["006",  "026", "090", "105", "108" ,"134", "150" ,"181"]
    # scenes = ["150"]
    # # scenes=["090"]
    # for scene in scenes: 
    #     dir_path="/media/ml4u/ExtremeSSD/datasets/waymo/processed/150"
    #     # dir_path="/media/ml4u/ExtremeSSD/datasets/log/TheHIep/StreetGaussian/output_deformable/waymo_exp/" + 'waymo_val_' + scene

    #     num_images = len(glob2.glob(os.path.join(dir_path, 'pseudo_depth_viz', '*_0.png'))) + len(glob2.glob(os.path.join(dir_path, 'test/ours_30000', '*_gt.png')))
    #     tmp = glob2.glob(os.path.join(dir_path, 'pseudo_depth_viz', '*_0.png'))
    #     tmp = [x.split('/')[-1] for x in tmp]
    #     tmp = [int(x.split('_')[0]) for x in tmp]
    #     min_indx = min(tmp)        

    #     for i in range(num_images):
    #         index = min_indx + i
    #         gt_img = cv2.imread(os.path.join(dir_path, 'images' ,f'{index:06d}_0.png'))
    #         gt_img = cv2.resize(gt_img, dsize=(1024, 512))

    #         depth_img = cv2.imread(os.path.join(dir_path, 'pseudo_depth_viz', f'{index:06d}_0.png'))
    #         depth_img = cv2.resize(depth_img, dsize=(1024, 512))
            
    #         out.write(np.hstack([gt_img, depth_img]))
    #         print(i)
    # out.release()
    
    out = cv2.VideoWriter('viz_pseudo_depth_midas.mp4',fourcc, 15.0, (1024*2, 512))
    scenes=["006",  "026", "090", "105", "108" ,"134", "150" ,"181"]
    # scenes=["006",  "026", "090", "105", "108" ,"134", "150" ,"181"]
    # scenes = ["150"]
    # scenes=["090"]
    for scene in scenes: 
        dir_path=f"/media/ml4u/ExtremeSSD/datasets/waymo/processed/{scene}"
        # dir_path="/media/ml4u/ExtremeSSD/datasets/log/TheHIep/StreetGaussian/output_deformable/waymo_exp/" + 'waymo_val_' + scene

        num_images = len(glob2.glob(os.path.join(dir_path, 'midas_depth', '*_0.png')))
        tmp = glob2.glob(os.path.join(dir_path, 'midas_depth', '*_0.png'))
        tmp = [x.split('/')[-1] for x in tmp]
        tmp = [int(x.split('_')[0]) for x in tmp]
        min_indx = min(tmp)        

        for i in range(num_images):
            index = min_indx + i
            gt_img = cv2.imread(os.path.join(dir_path, 'images' ,f'{index:06d}_0.png'))
            gt_img = cv2.resize(gt_img, dsize=(1024, 512))

            depth_img = cv2.imread(os.path.join(dir_path, 'midas_depth', f'{index:06d}_0.png'))
            depth_img = cv2.resize(depth_img, dsize=(1024, 512))
            
            out.write(np.hstack([gt_img, depth_img]))
            print(i)
    out.release()
    
    # out = cv2.VideoWriter('viz_pseudo_optical_flow.mp4',fourcc, 15.0, (1024*3, 512))
    # # scenes=["006",  "026", "090", "105", "108" ,"134", "150" ,"181"]
    # # scenes=["006",  "026", "090", "105", "108" ,"134", "150" ,"181"]
    # scenes = ["150"]
    # # scenes=["090"]
    # for scene in scenes: 
    #     dir_path="/media/ml4u/ExtremeSSD/datasets/waymo/processed/150"
    #     # dir_path="/media/ml4u/ExtremeSSD/datasets/log/TheHIep/StreetGaussian/output_deformable/waymo_exp/" + 'waymo_val_' + scene

    #     num_images = 100
    #     tmp = glob2.glob(os.path.join(dir_path, 'pseudo_optical_flow_viz', '*_0_flow.jpg'))
    #     tmp = [x.split('/')[-1] for x in tmp]
    #     tmp = [int(x.split('_')[0]) for x in tmp]
    #     min_indx = 96    

    #     for i in range(num_images-1):
    #         index = min_indx + i
    #         if True:
    #             gt_img = cv2.imread(os.path.join(dir_path, 'images' ,f'{index:06d}_0.png'))
    #             gt_img = cv2.resize(gt_img, dsize=(1024, 512))

    #             pred_depth_img = cv2.imread(os.path.join('/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/street_gaussian_wo_3DOD_mask_dynamic_object_and_pseudo_depth_filter_dynamic_lidar/fine_stage_train_only_dynamic_region_bounds_1600_embedding_feat_med_scale_depth_increase_embed_feat_lr_deformable_lr/waymo_exp/waymo_val_150/test/ours_30000', f'{index:06d}_0_flow.jpg'))
    #             pred_depth_img = cv2.resize(pred_depth_img, dsize=(1024, 512))
                
                
    #             depth_img = cv2.imread(os.path.join(dir_path, 'pseudo_optical_flow_viz', f'{index:06d}_0_flow.jpg'))
    #             depth_img = cv2.resize(depth_img, dsize=(1024, 512))
                
    #             out.write(np.hstack([gt_img, pred_depth_img, depth_img]))
    #             print(i)
    # out.release()