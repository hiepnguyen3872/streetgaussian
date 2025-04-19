################################################################################

import argparse
import os
import re
import cv2
import numpy as np
import time
from skimage.util import random_noise
import glob2
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim
import torch
from PIL import Image
from lib.utils.general_utils import PILtoTorch
if __name__ == "__main__": 


    scenes=["006",  "026", "090", "105", "108" ,"134", "150" ,"181"]
    # scenes = ["006"]
    # scenes=["090"]
    l1_test_total = 0.0
    psnr_test_total = 0.0
    ssim_test_total = 0.0
        
    l1_train_total = 0.0
    psnr_train_total = 0.0
    ssim_train_total = 0.0
    
    path_to_model_dir = "/home/ml4u/BKTeam/TheHiep/street_gaussians/output_street_gaussian_wo_3DOD/"
    # path_to_model_dir = "/home/ml4u/BKTeam/TheHiep/street_gaussians/output_semantic_warmup/"
    # path_to_model_dir = "/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/output_street_gaussian/"
    for scene in scenes: 
        dir_path=path_to_model_dir + 'waymo_exp/waymo_val_' + scene
        num_images = len(glob2.glob(os.path.join(dir_path, 'train/ours_30000', '*_gt.png'))) + len(glob2.glob(os.path.join(dir_path, 'test/ours_30000', '*_gt.png')))
        tmp = glob2.glob(os.path.join(dir_path, 'train/ours_30000', '*_gt.png'))
        tmp = [x.split('/')[-1] for x in tmp]
        tmp = [int(x.split('_')[0]) for x in tmp]
        min_indx = min(tmp)
        l1_test = 0.0
        psnr_test = 0.0
        ssim_test = 0.0
        
        l1_train = 0.0
        psnr_train = 0.0
        ssim_train = 0.0

        num_test = 0
        num_train = 0
        for i in range(num_images):
            tmp = None
            if i % 4 == 0 and i != 0: 
                image_path = os.path.join(dir_path, 'test/ours_30000')
                num_test += 1
                # image_path_static = os.path.join(dir_path, 'test_static/ours_30000')
            else: 
                image_path = os.path.join(dir_path, 'train/ours_30000')
                num_train += 1
                # image_path_static = os.path.join(dir_path, 'train_static/ours_30000')
            index = min_indx + i
            # image = cv2.imread(os.path.join(image_path, f'{index:06d}_0_rgb.png')) / 255.0  # Normalize to range 0-1
            # gt_image = cv2.imread(os.path.join(image_path, f'{index:06d}_0_gt.png')) / 255.0  # Normalize to range 0-1
            
            # image = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert to CxHxW
            # gt_image = torch.from_numpy(gt_image).permute(2, 0, 1).float()  # Convert to CxHxW
            image = Image.open(os.path.join(image_path, f'{index:06d}_0_rgb.png'))
            image = PILtoTorch(image, image.size, resize_mode=Image.BILINEAR)[:3, ...]
            
            gt_image = Image.open(os.path.join(image_path, f'{index:06d}_0_gt.png'))
            gt_image = PILtoTorch(gt_image, gt_image.size, resize_mode=Image.BILINEAR)[:3, ...]
            
            mask = torch.ones_like(gt_image[0]).bool()

            if i % 4 == 0 and i != 0: 
                l1_test += l1_loss(image, gt_image, mask).mean().double()
                psnr_test += psnr(image, gt_image, mask).mean().double()
            else:
                l1_train += l1_loss(image, gt_image, mask).mean().double()
                psnr_train += psnr(image, gt_image, mask).mean().double()

        psnr_test /= num_test
        l1_test /= num_test
        
        psnr_train /= num_train
        l1_train /= num_train
                
        log_file_path = os.path.join(dir_path, 'metrics_eval.txt')

        # Open the log file in append mode
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Test Num samples: {num_test},L1 Loss: {l1_test}, PSNR: {psnr_test}\n")
            log_file.write(f"Train Num samples: {num_train},L1 Loss: {l1_train}, PSNR: {psnr_train}\n")
        
        
        l1_test_total += l1_test
        psnr_test_total += psnr_test
            
        l1_train_total += l1_train
        psnr_train_total += psnr_train
    
    l1_test_total /= len(scenes)
    psnr_test_total /= len(scenes)
        
    l1_train_total /= len(scenes)
    psnr_train_total /= len(scenes)
    
    log_file_path_all_scenes = os.path.join(path_to_model_dir, 'metrics_eval_all_scenes.txt')

        # Open the log file in append mode
    with open(log_file_path_all_scenes, 'w') as log_file:
        log_file.write(f"Test Num scenes: {len(scenes)}, L1 Loss: {l1_test_total}, PSNR: {psnr_test_total}\n")
        log_file.write(f"Train Num scenes: {len(scenes)}, L1 Loss: {l1_train_total}, PSNR: {psnr_train_total}\n")
        
    


    