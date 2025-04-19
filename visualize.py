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
    segmentation_dir = '/media/ml4u/ExtremeSSD/datasets/waymo/processed'

    out = cv2.VideoWriter('viz_video/viz_gen_3_pseudo_views_lifespan_colmap.mp4',fourcc, 7.0, (1024*4, 512*5))
    # scenes=["006",  "026", "090", "105", "108" ,"134", "150" ,"181"]
    # scenes=["006",  "026", "090", "105", "108" ,"134", "150" ,"181"]
    scenes = ["150"]
    # # scenes=["090"]
    for scene in scenes: 
    #     dir_path="/media/ml4u/ExtremeSSD/datasets/log/TheHIep/StreetGaussian/output/waymo_exp/" + 'waymo_val_' + scene
    #     # dir_path="/media/ml4u/ExtremeSSD/datasets/log/TheHIep/StreetGaussian/output_deformable/waymo_exp/" + 'waymo_val_' + scene

    #     num_images = len(glob2.glob(os.path.join(dir_path, 'train/ours_30000', '*_gt.png'))) + len(glob2.glob(os.path.join(dir_path, 'test/ours_30000', '*_gt.png')))
    #     tmp = glob2.glob(os.path.join(dir_path, 'train/ours_30000', '*_gt.png'))
    #     tmp = [x.split('/')[-1] for x in tmp]
    #     tmp = [int(x.split('_')[0]) for x in tmp]
    #     min_indx = min(tmp)
        

    #     for i in range(num_images):
    #         if i % 4 == 0 and i != 0: 
    #             image_path = os.path.join(dir_path, 'test/ours_30000')
    #             # image_path_static = os.path.join(dir_path, 'test_static/ours_30000')
    #         else: 
    #             image_path = os.path.join(dir_path, 'train/ours_30000')
    #             # image_path_static = os.path.join(dir_path, 'train_static/ours_30000')
    #         index = min_indx + i
    #         render_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_rgb.png'))
    #         render_img = cv2.resize(render_img, dsize=(1024, 512))

    #         gt_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_gt.png'))
    #         gt_img = cv2.resize(gt_img, dsize=(1024, 512))

    #         depth_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_depth.png'))
    #         depth_img = cv2.resize(depth_img, dsize=(1024, 512))

    #         # depth_img = cv2.imread(os.path.join(image_path_static, f'{index:06d}_0_rgb.png'))
    #         # depth_img = cv2.resize(depth_img, dsize=(1024, 512))

    #         out.write(np.hstack([gt_img, render_img, depth_img]))
    #         print(i)
    # out.release()




        # dir_paths=["/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/output_street_gaussian/waymo_exp/"+ 'waymo_val_' + scene,
        #            "/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/street_gaussian_wo_3DOD_mask_dynamic_object_and_pseudo_depth_filter_dynamic_lidar/fine_train_only_dynamic_region_bounds_1600_embedding_feats/waymo_exp/" + "waymo_val_" + scene,
        #            "/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/street_gaussian_wo_3DOD_mask_dynamic_object_and_pseudo_depth_filter_dynamic_lidar/fine_train_only_dynamic_region_bounds_1600_embedding_feat_med_scale_depth_use_large_adjoin_regular_large_lambda_dx/waymo_exp/" + "waymo_val_" + scene,
        #            "/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/street_gaussian_wo_3DOD_mask_dynamic_object_and_pseudo_depth_filter_dynamic_lidar/fine_train_only_dynamic_region_bounds_1600_embedding_feat_med_scale_depth_use_large_adjoin_regular_large_lambda_dx_use_time_emb/waymo_exp/" + "waymo_val_" + scene,
        #            ]
    #     dir_paths = ["/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/street_gaussian_wo_3DOD_mask_dynamic_object_and_pseudo_depth_filter_dynamic_lidar/pseudo_view_loss_with_warp_gt_image_t1_test_midas/waymo_exp/" + "waymo_val_" + scene]
    #     # model_name = ["3DGS Original", "Add Embedding Feats", "Add Embedding Feats + Deformation Regularization", "Add Embedding Feats + Deformation Regularization + Time Embedding"]
    #     model_name = [""]
    #     num_images = len(glob2.glob(os.path.join(dir_paths[0], 'train/ours_30000', '*_gt.png'))) + len(glob2.glob(os.path.join(dir_paths[0], 'test/ours_30000', '*_gt.png')))
    #     tmp = glob2.glob(os.path.join(dir_paths[0], 'train/ours_30000', '*_gt.png'))
    #     tmp = [x.split('/')[-1] for x in tmp]
    #     tmp = [int(x.split('_')[0]) for x in tmp]
    #     min_indx = min(tmp)
    #     for i in range(num_images):
    #         tmp = None
    #         for j, dir_path in enumerate(dir_paths):
    #             if i % 4 == 0 and i != 0: 
    #                 image_path = os.path.join(dir_path, 'test/ours_30000')
    #                 # image_path_static = os.path.join(dir_path, 'test_static/ours_30000')
    #             else: 
    #                 image_path = os.path.join(dir_path, 'train/ours_30000')
    #                 # image_path_static = os.path.join(dir_path, 'train_static/ours_30000')
    #             index = min_indx + i
    #             render_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_rgb.png'))
    #             render_img = cv2.resize(render_img, dsize=(1024, 512))
    #             gt_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_gt.png'))
    #             gt_img = cv2.resize(gt_img, dsize=(1024, 512))
    #             # Add text 'scene' to gt_img
    #             cv2.putText(gt_img, f'Scene: {scene}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    #             render_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_rgb.png'))
    #             render_img = cv2.resize(render_img, dsize=(1024, 512))
    #             # Add text 'model' to render_img
    #             cv2.putText(render_img, model_name[j], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    #             if index == 97: 
    #                 continue
    #             # depth_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_depth.png'))
    #             # depth_img = cv2.resize(depth_img, dsize=(1024, 512))
    #             # if j == 0: 
    #             #     seg_img = cv2.imread(os.path.join(segmentation_dir, scene, 'pseudo_seg_colors', f'{index:06d}_0.png'))
    #             #     seg_img = cv2.resize(seg_img, dsize=(1024, 512))
    #             # else:
    #             #     seg_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_semantic.png'))
    #             #     seg_img = cv2.resize(seg_img, dsize=(1024, 512))
    #             dx_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_dx_new.png'))
    #             dx_img = cv2.resize(dx_img, dsize=(1024, 512))
    #             if tmp is None:
    #                 tmp = np.hstack([gt_img, render_img, dx_img])
    #             else: 
    #                 tmp = np.vstack([tmp, np.hstack([gt_img, render_img, dx_img])])
    #         out.write(tmp)
    #         print(index)
    # out.release()
    
    
    
    

    #     dir_paths = ["/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/street_gaussian_wo_3DOD_mask_dynamic_object_and_pseudo_depth_filter_dynamic_lidar/pseudo_view_loss_with_warp_gt_image_t1_test_midas/waymo_exp/" + "waymo_val_" + scene]
    #     num_images = len(glob2.glob(os.path.join(dir_paths[0], 'train/ours_30000', '*_gt.png'))) + len(glob2.glob(os.path.join(dir_paths[0], 'test/ours_30000', '*_gt.png')))
    #     tmp = glob2.glob(os.path.join(dir_paths[0], 'train/ours_30000', '*_gt.png'))
    #     tmp = [x.split('/')[-1] for x in tmp]
    #     tmp = [int(x.split('_')[0]) for x in tmp]
    #     min_indx = min(tmp)
    #     for i in range(num_images):
    #         tmp = None
    #         for dir_path in dir_paths:
    #             if i % 4 == 0 and i != 0: 
    #                 image_path = os.path.join(dir_path, 'test/ours_30000')
    #                 # image_path_static = os.path.join(dir_path, 'test_static/ours_30000')
    #             else: 
    #                 image_path = os.path.join(dir_path, 'train/ours_30000')
    #                 # image_path_static = os.path.join(dir_path, 'train_static/ours_30000')
    #             index = min_indx + i
                
    #             render_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_rgb.png'))
    #             render_img = cv2.resize(render_img, dsize=(1024, 512))
                
    #             gt_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_gt.png'))
    #             gt_img = cv2.resize(gt_img, dsize=(1024, 512))
                
    #             depth_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_depth.png'))
    #             depth_img = cv2.resize(depth_img, dsize=(1024, 512))
                
    #             semantic_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_semantic.png'))
    #             semantic_img = cv2.resize(semantic_img, dsize=(1024, 512))
    #             # depth_img = cv2.imread(os.path.join(image_path_static, f'{index:06d}_0_rgb.png'))
    #             # depth_img = cv2.resize(depth_img, dsize=(1024, 512))
    #             tmp = np.vstack([np.hstack([gt_img, render_img]), np.hstack([depth_img, semantic_img])])
    #         out.write(tmp)
    #         print(i)
    # out.release()

    
    
        dir_paths=["/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/output_street_gaussian/waymo_exp/"+ 'waymo_val_' + scene,
                   "/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/street_gaussian_wo_3DOD_mask_dynamic_object_and_pseudo_depth_filter_dynamic_lidar/fine_train_only_dynamic_region_bounds_1600_embedding_feats/waymo_exp/" + "waymo_val_" + scene,
                   "/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/street_gaussian_wo_3DOD_mask_dynamic_object_and_pseudo_depth_filter_dynamic_lidar/pseudo_3_view_loss_with_warp_gt_image_combined_threshold_0.05_60000_iteration/waymo_exp/" + "waymo_val_" + scene,
                   "/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/street_gaussian_wo_3DOD_mask_dynamic_object_and_pseudo_depth_filter_dynamic_lidar/pseudo_3_view_loss_with_warp_gt_image_combined_threshold_0.05_60000_iteration_lifespan/waymo_exp/" + "waymo_val_" + scene,
                   "/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/street_gaussian_wo_3DOD_mask_dynamic_object_and_pseudo_depth_filter_dynamic_lidar/pseudo_3_view_loss_with_warp_gt_image_combined_threshold_0.05_60000_iteration_lifespan_colmap/waymo_exp/" + "waymo_val_" + scene,
                   ]
        model_name = ["3DGS Original", "feature embedding", "gen 3 pseudo views", "lifespan", "lifespan + colmap"]
        num_images = len(glob2.glob(os.path.join(dir_paths[0], 'train/ours_30000', '*_gt.png'))) + len(glob2.glob(os.path.join(dir_paths[0], 'test/ours_30000', '*_gt.png')))
        tmp = glob2.glob(os.path.join(dir_paths[0], 'train/ours_30000', '*_gt.png'))
        tmp = [x.split('/')[-1] for x in tmp]
        tmp = [int(x.split('_')[0]) for x in tmp]
        min_indx = min(tmp)
        for i in range(num_images):
            tmp = None
            for j, dir_path in enumerate(dir_paths):
                if i % 4 == 0 and i != 0: 
                    image_path = os.path.join(dir_path, 'test/ours_30000')
                    if '60000' in dir_path: 
                        image_path = os.path.join(dir_path, 'test/ours_60000')
                    if '90000' in dir_path: 
                        image_path = os.path.join(dir_path, 'test/ours_90000')
                    # image_path_static = os.path.join(dir_path, 'test_static/ours_30000')
                else: 
                    image_path = os.path.join(dir_path, 'train/ours_30000')
                    if '60000' in dir_path: 
                        image_path = os.path.join(dir_path, 'train/ours_60000')
                    if '90000' in dir_path: 
                        image_path = os.path.join(dir_path, 'train/ours_90000')
                    # image_path_static = os.path.join(dir_path, 'train_static/ours_30000')
                index = min_indx + i
                render_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_rgb.png'))
                render_img = cv2.resize(render_img, dsize=(1024, 512))
                gt_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_gt.png'))
                gt_img = cv2.resize(gt_img, dsize=(1024, 512))
                # Add text 'scene' to gt_img
                cv2.putText(gt_img, f'Scene: {scene}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

                render_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_rgb.png'))
                render_img = cv2.resize(render_img, dsize=(1024, 512))
                # Add text 'model' to render_img
                cv2.putText(render_img, model_name[j], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                if index == 97: 
                    continue
                depth_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_depth.png'))
                depth_img = cv2.resize(depth_img, dsize=(1024, 512))
                if j == 0: 
                    seg_img = cv2.imread(os.path.join(segmentation_dir, scene, 'pseudo_seg_colors', f'{index:06d}_0.png'))
                    seg_img = cv2.resize(seg_img, dsize=(1024, 512))
                else:
                    seg_img = cv2.imread(os.path.join(image_path, f'{index:06d}_0_semantic.png'))
                    seg_img = cv2.resize(seg_img, dsize=(1024, 512))
                if tmp is None:
                    tmp = np.hstack([gt_img, render_img, depth_img, seg_img])
                else: 
                    tmp = np.vstack([tmp, np.hstack([gt_img, render_img, depth_img, seg_img])])
            out.write(tmp)
            print(index)
    out.release()