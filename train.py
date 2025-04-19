import os
import torch
from random import randint
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim, silog_loss, get_dynamic_mask, smooth_loss, warp_reconstruction_loss, save_img_torch, visualize_depth, temporal_smoothness_knn
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy, visualize_segmentation, visualize_dx
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.general_utils import safe_state, o3d_knn
from lib.utils.camera_utils import Camera
from lib.utils.cfg_utils import save_cfg
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.config import cfg
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from lib.utils.system_utils import searchForMaxIteration
import time
import numpy as np
import random
from lib.utils.optical_flow_utils import calculate_gs_optical_flow

import matplotlib.pyplot as plt
from PIL import Image


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables

def scene_reconstruction(gaussians, scene, gaussians_renderer, stage, tb_writer):
    start_iter = 0
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_dict = {}
    progress_bar = tqdm(range(start_iter, training_args.iterations))
    start_iter += 1

    viewpoint_stack = None
    
    # Get viewpoint_cam in order to train instead of random
    current_camera_index = 0
    iteration = start_iter
    # while iteration <= training_args.iterations:
    iteration_thres_dynamic = 0
    
    pseudo_viewpoint_stack = None
    for iteration in range(start_iter, training_args.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # if iteration > training_args.iterations and not cfg.model.nsg.use_deformation_model:
        # if iteration > 5000 and not cfg.model.nsg.use_deformation_model:
        #     iteration = start_iter
        #     cfg.model.nsg.use_deformation_model = use_deformation_model
        #     stage = "fine"

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # iteration += 1

        # Every 1000 iterations upsample
        # if iteration % 1000 == 0:
        #     if resolution_scales:  
        #         scale = resolution_scales.pop()
        
        # Train with pseudo view
        # if iteration % 3 == 0 and cfg.data.generate_pseudo_view: 
        if cfg.data.generate_pseudo_view: 
            if not pseudo_viewpoint_stack:
                pseudo_viewpoint_stack = scene.getPseudoTrainCameras().copy()
            
            ## Visualize image generated from pseudo view t'
            # for i in range(len(pseudo_viewpoint_stack)):
            #     image_np = np.transpose(pseudo_viewpoint_stack[i].original_image.cpu().numpy(), (1, 2, 0))

            #     mask = pseudo_viewpoint_stack[i].original_mask.cpu().numpy()
            #     mask_np = np.transpose(mask, (1, 2, 0))
            #     mask_np = np.repeat(mask_np, 3, axis=2)
                            
            #     concatenated_np = np.concatenate((image_np, mask_np), axis=1)

            #     # Visualize the concatenated image using matplotlib
            #     plt.imshow(concatenated_np)
            #     plt.axis('off')  # Turn off axis labels
            #     plt.show()

            #     # Save the concatenated image using PIL
            #     concatenated_pil = Image.fromarray((concatenated_np * 255).astype(np.uint8))
            #     concatenated_pil.save(f'viz_video/pseudo_view/combined_sky_mask_diff_threshold_0.3/concatenated_output_image_{i}.png')

            # breakpoint()
            
            
            pseudo_viewpoint_cam: Camera = pseudo_viewpoint_stack.pop(randint(0, len(pseudo_viewpoint_stack) - 1))
            # pseudo_render_pkg = gaussians_renderer.render(pseudo_viewpoint_cam, gaussians, return_dx=False, use_hexplane=use_hexplane)
            # reconstruct_loss, warp_reconstruct_img, depth_t1 = warp_reconstruction_loss(pseudo_viewpoint_cam, pseudo_render_pkg, lambda_l1=optim_args.lambda_l1, lambda_dssim=optim_args.lambda_dssim)
            # scalar_dict['pseudo_view_reconstruct_loss'] = reconstruct_loss
            # loss = reconstruct_loss
            # os.makedirs(f"{cfg.model_path}/log_images", exist_ok = True)
            # if iteration % 100 == 0:
            #     save_img_torch(warp_reconstruct_img, os.path.join(f"{cfg.model_path}/log_images/{iteration}_warp_reconstruct.png"))
            #     visualize_depth(depth_t1, os.path.join(f"{cfg.model_path}/log_images/{iteration}_depth.png"))
            # scalar_dict['loss'] = loss.item()

            # loss.backward()
            
            # iter_end.record()
            
            # continue

        
            
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        camera_id = randint(0, len(viewpoint_stack)-1)
        # viewpoint_cam: Camera = viewpoint_stack.pop(camera_id)
        viewpoint_cam: Camera = viewpoint_stack[camera_id]
        if camera_id < len(viewpoint_stack) - 2: 
            viewpoint_cam_next = viewpoint_stack[camera_id + 1]
        else: 
            viewpoint_cam_next = None
        
        # # Get viewpoint_came in order to train instead of random
        # if current_camera_index >= len(viewpoint_stack):
        #     current_camera_index = 0  # Reset to the beginning if it's gone through all cameras        
        # # current_camera_index = iteration // (training_args.iterations // len(viewpoint_stack))
        # # if current_camera_index >= len(viewpoint_stack): 
        # #     current_camera_index = -1
        # if iteration <= iteration_thres_dynamic and cfg.model.nsg.train_key_frame: 
        #     # current_camera_index = random.choice(list(range(0, len(viewpoint_stack), 30)))
        #     current_camera_index = 0
        # viewpoint_cam: Camera = viewpoint_stack[current_camera_index]

        # current_camera_index += 1
        # ====================================================================
        # Get mask
        # original_mask: pixel in original_mask with 0 will not be surpervised
        # original_acc_mask: use to suepervise the acc result of rendering
        # original_sky_mask: sky mask

        gt_image = viewpoint_cam.original_image.cuda()
        if hasattr(viewpoint_cam, 'original_mask'):
            mask = viewpoint_cam.original_mask.cuda().bool()
        else:
            mask = torch.ones_like(gt_image[0:1]).bool()
        
        if cfg.model.mask_2d_dynamic_object: 
            dynamic_mask = get_dynamic_mask(viewpoint_cam.meta['semantic'].cuda().long())
            if stage == 'fine' and iteration >= iteration_thres_dynamic: 
                dynamic_mask = ~dynamic_mask
            # else:
            #     dynamic_mask = torch.zeros_like(gt_image[0:1]).bool()
            
            if stage == 'coarse':
                mask = mask & ~dynamic_mask
        if hasattr(viewpoint_cam, 'original_sky_mask'):
            sky_mask = viewpoint_cam.original_sky_mask.cuda()
        else:
            sky_mask = None
            
        if hasattr(viewpoint_cam, 'original_obj_bound'):
            obj_bound = viewpoint_cam.original_obj_bound.cuda().bool()
        else:
            obj_bound = torch.zeros_like(gt_image[0:1]).bool()
        
        if (iteration - 1) == training_args.debug_from:
            cfg.render.debug = True
        
        if iteration >= iteration_thres_dynamic: 
            use_hexplane = True
        else: 
            use_hexplane = False
        # if iteration >= 10000: 
        #     only_backprop_hexplane = True
        # else: 
        #     only_backprop_hexplane = False
            
        # render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians, return_dx=cfg.model.nsg.use_deformation_model, use_hexplane=use_hexplane, only_backprop_hexplane = only_backprop_hexplane)
        render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians, return_dx=cfg.model.nsg.use_deformation_model, use_hexplane=use_hexplane)
        image, acc, viewspace_point_tensor, visibility_filter, radii = render_pkg["rgb"], render_pkg['acc'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg['depth'] # [1, H, W]
        scalar_dict = dict()
        # rgb loss
        Ll1 = l1_loss(image, gt_image, mask)
        scalar_dict['l1_loss'] = Ll1.item()
        loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask))
        
        ## regularization for time center and time variation
        gaussian_dynamic_mask = render_pkg['gaussian_dynamic_mask']
        if cfg.model.nsg.use_deformation_model and len(gaussian_dynamic_mask[gaussian_dynamic_mask]) > 10: 
            L_time_center, L_time_variation = temporal_smoothness_knn(render_pkg['means3D'][gaussian_dynamic_mask].detach(), render_pkg['mu_t'], render_pkg['sigma_t'], k=10)
            scalar_dict['l_time_center'] = L_time_center.item()
            scalar_dict['l_time_variation'] = L_time_variation.item()
            loss += L_time_center
            loss += L_time_variation

            # choose margin = 0.5
            L_time_variation_reg = torch.clamp(render_pkg['sigma_t'] - 0.5, min=0).mean()
            scalar_dict['l_time_variation_reg'] = L_time_variation_reg.item()
            loss += 0.01*L_time_variation_reg

        ## Regularization loss for velocity and accelerator prediction
        # scalar_dict['da_dv_loss'] = render_pkg["da_dv_loss"].item()
        # scalar_dict['smooth_dv_loss'] = render_pkg["smooth_dv_loss"].item()
        # scalar_dict['smooth_da_loss'] = render_pkg["smooth_da_loss"].item()
        # loss += render_pkg["da_dv_loss"] + render_pkg["smooth_dv_loss"] + render_pkg["smooth_da_loss"]

        # if viewpoint_cam_next is not None: 
        #     render_pkg_t1 = gaussians_renderer.render(viewpoint_cam, gaussians, return_dx=cfg.model.nsg.use_deformation_model, use_hexplane=use_hexplane, only_backprop_hexplane=True)
        #     render_pkg_t2 = gaussians_renderer.render(viewpoint_cam_next, gaussians, return_dx=cfg.model.nsg.use_deformation_model, use_hexplane=use_hexplane, only_backprop_hexplane=True)
        #     gs_optical_flow = calculate_gs_optical_flow(render_pkg_t1, render_pkg_t2)
        #     pseudo_optical_flow = viewpoint_cam.meta['pseudo_optical_flow'].permute(0, 2, 3, 1).squeeze()
        #     pseudo_optical_flow = pseudo_optical_flow.cuda()
        #     large_motion_msk = torch.norm(pseudo_optical_flow, p=2, dim=-1) >= 1.  # flow_thresh = 0.1 or other value to filter out noise, here we assume that we have already loaded pre-computed optical flow somewhere as pseudo GT
        #     Lflow = torch.norm((pseudo_optical_flow - gs_optical_flow)[large_motion_msk], p=2, dim=-1).mean() 
        #     loss += 0.001 * Lflow
        #     scalar_dict['pseudo_optical_flow_loss'] = Lflow

        # Pseudo view Loss
        if cfg.data.generate_pseudo_view: 
            pseudo_render_pkg = gaussians_renderer.render(pseudo_viewpoint_cam, gaussians, return_dx=False, use_hexplane=use_hexplane)
            # reconstruct_loss, source_img, target_img, warp_reconstruct_img, depth_t1 = warp_reconstruction_loss(pseudo_viewpoint_cam, pseudo_render_pkg, lambda_l1=optim_args.lambda_l1, lambda_dssim=optim_args.lambda_dssim)
            # scalar_dict['pseudo_view_reconstruct_loss'] = reconstruct_loss
            # loss += reconstruct_loss
            pseudo_Ll1 = l1_loss(pseudo_render_pkg['rgb'], pseudo_viewpoint_cam.original_image.cuda(), pseudo_viewpoint_cam.original_mask)
            pseudo_loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * pseudo_Ll1 + optim_args.lambda_dssim * (1.0 - ssim(pseudo_render_pkg['rgb'], pseudo_viewpoint_cam.original_image.cuda(), mask=pseudo_viewpoint_cam.original_mask.cuda()))
            scalar_dict['pseudo_loss'] = pseudo_loss
            loss += pseudo_loss
        # dx loss
        # No need when filter dynamic class gaussians
        if cfg.model.nsg.use_deformation_model:
            if render_pkg['dx'] is not None:
                dx_abs = torch.abs(render_pkg['dx'])
                dx_loss = torch.mean(dx_abs) * optim_args.deformable.lambda_dx
                scalar_dict['dx_loss'] = dx_loss
                loss += dx_loss

        # dshs loss
            if render_pkg['dshs'] is not None:
                dshs_abs = torch.abs(render_pkg['dshs'])
                dshs_loss = torch.mean(dshs_abs) * optim_args.deformable.lambda_dshs
                scalar_dict['dshs_loss'] = dshs_loss
                loss += dshs_loss
        # sky loss
        if optim_args.lambda_sky > 0 and gaussians.include_sky and sky_mask is not None:
            acc = torch.clamp(acc, min=1e-6, max=1.-1e-6)
            sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
            if len(optim_args.lambda_sky_scale) > 0:
                sky_loss *= optim_args.lambda_sky_scale[viewpoint_cam.meta['cam']]
            scalar_dict['sky_loss'] = sky_loss.item()
            loss += optim_args.lambda_sky * sky_loss

        # semantic loss
        if optim_args.lambda_semantic > 0 and data_args.get('use_semantic', False) and 'semantic' in viewpoint_cam.meta:
            gt_semantic = viewpoint_cam.meta['semantic'].cuda().long() # [1, H, W]
            if torch.all(gt_semantic == -1):
                semantic_loss = torch.zeros_like(Ll1)
            else:
                semantic = render_pkg['semantic'].unsqueeze(0) # [1, S, H, W]
                semantic_loss = torch.nn.functional.cross_entropy(
                    input=semantic, 
                    target=gt_semantic,
                    ignore_index=-1, 
                    reduction='none'
                )
            masked_semantic_loss = semantic_loss[mask]  # Only keep losses where mask is True
            semantic_loss = masked_semantic_loss.mean()
            scalar_dict['semantic_loss'] = semantic_loss.item()
            loss += optim_args.lambda_semantic * semantic_loss
        
        if optim_args.lambda_reg > 0 and gaussians.include_obj and iteration >= optim_args.densify_until_iter:
            render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
            image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = torch.clamp(acc_obj, min=1e-6, max=1.-1e-6)

            # box_reg_loss = gaussians.get_box_reg_loss()
            # scalar_dict['box_reg_loss'] = box_reg_loss.item()
            # loss += optim_args.lambda_reg * box_reg_loss

            obj_acc_loss = torch.where(obj_bound, 
                -(acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj)), 
                -torch.log(1. - acc_obj)).mean()
            scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
            loss += optim_args.lambda_reg * obj_acc_loss
            # obj_acc_loss = -((acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj))).mean()
            # scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
            # loss += optim_args.lambda_reg * obj_acc_loss
        
        # lidar depth loss
        if optim_args.lambda_depth_lidar > 0 and 'lidar_depth' in viewpoint_cam.meta:            
            lidar_depth = viewpoint_cam.meta['lidar_depth'].cuda() # [1, H, W]
            depth_mask = torch.logical_and((lidar_depth > 0.), mask)
            # depth_mask[obj_bound] = False
            if torch.nonzero(depth_mask).any():
                expected_depth = depth / (render_pkg['acc'] + 1e-10)  
                depth_error = torch.abs((expected_depth[depth_mask] - lidar_depth[depth_mask]))
                depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
                lidar_depth_loss = depth_error.mean()
                scalar_dict['lidar_depth_loss'] = lidar_depth_loss
            else:
                lidar_depth_loss = torch.zeros_like(Ll1)  
            loss += optim_args.lambda_depth_lidar * lidar_depth_loss

        # if 'pseudo_depth' in viewpoint_cam.meta and cfg.model.deformable.smooth_render_dx_loss: 
        #     loss += smooth_loss(render_pkg['render_dx'], viewpoint_cam.meta['pseudo_depth'].cuda())
        
        # # pseudo depth loss
        # if optim_args.lambda_depth_lidar > 0 and 'pseudo_depth' in viewpoint_cam.meta:            
        #     pseudo_depth = viewpoint_cam.meta['pseudo_depth'].cuda() # [1, H, W]
        #     pseudo_depth = (pseudo_depth * lidar_depth[depth_mask].median()) / pseudo_depth.median()
        #     pseudo_depth_loss = silog_loss(depth[mask], pseudo_depth[mask])
        #     scalar_dict['pseudo_depth_loss'] = pseudo_depth_loss
        #     loss += optim_args.lambda_depth_lidar * pseudo_depth_loss
        # Constraint Depth Loss for non-dynamic_region
        if stage == 'fine' and cfg.model.mask_2d_dynamic_object and iteration >= iteration_thres_dynamic: 
            render_dynamic_mask = get_dynamic_mask(render_pkg['semantic_d'].argmax(axis=0).unsqueeze(0).cuda().long())
            miss_render_dynamic_mask = render_dynamic_mask & dynamic_mask
            if miss_render_dynamic_mask.sum() > 0:
                non_dynamic_depth_loss = torch.mean(render_pkg['depth_d'][render_dynamic_mask & dynamic_mask])
                loss +=  non_dynamic_depth_loss                    
        # color correction loss
        if optim_args.lambda_color_correction > 0 and gaussians.use_color_correction:
            color_correction_reg_loss = gaussians.color_correction.regularization_loss(viewpoint_cam)
            scalar_dict['color_correction_reg_loss'] = color_correction_reg_loss.item()
            loss += optim_args.lambda_color_correction * color_correction_reg_loss
        
        # pose correction loss
        if optim_args.lambda_pose_correction > 0 and gaussians.use_pose_correction:
            pose_correction_reg_loss = gaussians.pose_correction.regularization_loss()
            scalar_dict['pose_correction_reg_loss'] = pose_correction_reg_loss.item()
            loss += optim_args.lambda_pose_correction * pose_correction_reg_loss
                    
        # scale flatten loss
        if optim_args.lambda_scale_flatten > 0:
            scale_flatten_loss = gaussians.background.scale_flatten_loss()
            scalar_dict['scale_flatten_loss'] = scale_flatten_loss.item()
            loss += optim_args.lambda_scale_flatten * scale_flatten_loss
        
        # opacity sparse loss
        if optim_args.lambda_opacity_sparse > 0:
            opacity = gaussians.get_opacity
            opacity = opacity.clamp(1e-6, 1-1e-6)
            log_opacity = opacity * torch.log(opacity)
            log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
            sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()
            scalar_dict['opacity_sparse_loss'] = sparse_loss.item()
            loss += optim_args.lambda_opacity_sparse * sparse_loss
                
        # normal loss
        if optim_args.lambda_normal_mono > 0 and 'mono_normal' in viewpoint_cam.meta and 'normals' in render_pkg:
            if sky_mask is None:
                normal_mask = mask
            else:
                normal_mask = torch.logical_and(mask, ~sky_mask)
                normal_mask = normal_mask.squeeze(0)
                normal_mask[:50] = False
                
            normal_gt = viewpoint_cam.meta['mono_normal'].permute(1, 2, 0).cuda() # [H, W, 3]
            R_c2w = viewpoint_cam.world_view_transform[:3, :3]
            normal_gt = torch.matmul(normal_gt, R_c2w.T) # to world space
            normal_pred = render_pkg['normals'].permute(1, 2, 0) # [H, W, 3]    
            
            normal_l1_loss = torch.abs(normal_pred[normal_mask] - normal_gt[normal_mask]).mean()
            normal_cos_loss = (1. - torch.sum(normal_pred[normal_mask] * normal_gt[normal_mask], dim=-1)).mean()
            scalar_dict['normal_l1_loss'] = normal_l1_loss.item()
            scalar_dict['normal_cos_loss'] = normal_cos_loss.item()
            normal_loss = normal_l1_loss + normal_cos_loss
            loss += optim_args.lambda_normal_mono * normal_loss
            
        scalar_dict['loss'] = loss.item()

        loss.backward()
        
        iter_end.record()
                
        is_save_images = True
        if is_save_images and (iteration % 1000 == 0):
            # row0: gt_image, image, depth
            # row1: acc, image_obj, acc_obj
            # if depth.max() == 0: 
            #     print("depth value is all 0")
            #     continue
            depth_colored, _ = visualize_depth_numpy(depth.detach().cpu().numpy().squeeze(0))
            depth_colored = depth_colored[..., [2, 1, 0]] / 255.
            depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()
            row0 = torch.cat([gt_image, image, depth_colored], dim=2)
            acc = acc.repeat(3, 1, 1)
            with torch.no_grad():
                render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
                image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = acc_obj.repeat(3, 1, 1)
            row1 = torch.cat([acc, image_obj, acc_obj], dim=2)
            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
            os.makedirs(f"{cfg.model_path}/log_images", exist_ok = True)
            save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{iteration}.jpg")
            if cfg.data.use_semantic: 
                # Get semantic prediction (batch_size, num_classes, H, W) -> (H, W)
                semantic_pred = semantic.detach().cpu().numpy().argmax(axis=1).squeeze().astype(np.uint8)
                semantic_colored = visualize_segmentation(semantic_pred)
                semantic_colored.save(os.path.join(f"{cfg.model_path}/log_images/{iteration}_seg.png"))
            
            if cfg.model.deformable.viz_dx: 
                render_dx = render_pkg['render_dx'].detach().cpu()
                image_render_dx = visualize_dx(render_dx)
                image_render_dx.save(os.path.join(f"{cfg.model_path}/log_images/{iteration}_dx.png"))
            
            # if cfg.data.generate_pseudo_view:
            #     save_img_torch(source_img, os.path.join(f"{cfg.model_path}/log_images/{iteration}_source.png"))
            #     save_img_torch(target_img, os.path.join(f"{cfg.model_path}/log_images/{iteration}_target.png"))
            #     save_img_torch(warp_reconstruct_img, os.path.join(f"{cfg.model_path}/log_images/{iteration}_warp_reconstruct.png"))
            #     visualize_depth(depth_t1, os.path.join(f"{cfg.model_path}/log_images/{iteration}_depth.png"))
                
        torch.cuda.empty_cache()
        with torch.no_grad():
            # Log
            tensor_dict = dict()

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * ema_psnr_for_log
            if viewpoint_cam.id not in psnr_dict:
                psnr_dict[viewpoint_cam.id] = psnr(image, gt_image, mask).mean().float()
            else:
                psnr_dict[viewpoint_cam.id] = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * psnr_dict[viewpoint_cam.id]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Exp": f"{cfg.task}-{cfg.exp_name}", 
                                          "Loss": f"{ema_loss_for_log:.{7}f},", 
                                          "PSNR": f"{ema_psnr_for_log:.{4}f}",
                                         "Stage": stage})
                progress_bar.update(10)
            if iteration == training_args.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in training_args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < optim_args.densify_until_iter:
                # if not (cfg.model.nsg.train_key_frame and iteration > iteration_thres_dynamic):
                if True:
                    gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))
                    gaussians.parse_camera(viewpoint_cam) 
                    # if stage == "fine" and cfg.model.mask_2d_dynamic_object and iteration >= iteration_thres_dynamic: 
                    #     gaussians.set_max_radii2D(radii, visibility_filter, render_pkg['gaussian_dynamic_mask'])
                    #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg['gaussian_dynamic_mask'])
                    # else: 
                    gaussians.set_max_radii2D(radii, visibility_filter)
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    
                    prune_big_points = iteration > optim_args.opacity_reset_interval

                    if iteration > optim_args.densify_from_iter:
                        if iteration % optim_args.densification_interval == 0:
                            scalars, tensors = gaussians.densify_and_prune(
                                max_grad=optim_args.densify_grad_threshold,
                                min_opacity=optim_args.min_opacity,
                                prune_big_points=prune_big_points,
                            )

                            scalar_dict.update(scalars)
                            tensor_dict.update(tensors)
                        
            # Reset opacity
            if iteration < optim_args.densify_until_iter:
                # if not (cfg.model.nsg.train_key_frame and iteration > iteration_thres_dynamic):
                # if iteration < 15000:
                if True:
                    if iteration % optim_args.opacity_reset_interval == 0:
                        gaussians.reset_opacity()
                    if data_args.white_background and iteration == optim_args.densify_from_iter:
                        gaussians.reset_opacity()

            training_report(tb_writer, iteration, scalar_dict, tensor_dict, training_args.test_iterations, scene, gaussians_renderer)

            # Optimizer step
            if iteration < training_args.iterations:
                gaussians.update_optimizer()
            if (iteration in training_args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                state_dict = gaussians.save_state_dict(is_final=(iteration == training_args.iterations))
                state_dict['iter'] = iteration
                ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')
                torch.save(state_dict, ckpt_path)
                if cfg.model.nsg.use_deformation_model: 
                    gaussians.background.save_deformation(cfg.trained_model_dir, iteration)
            

def training():
    start_iter = 0
    tb_writer = prepare_output_and_logger()
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)
    if cfg.pretrain_model_path:
        assert os.path.exists(cfg.pretrain_model_path), f"{cfg.pretrain_model_path} does not exist"
        pretrain_model_dir = os.path.join(cfg.pretrain_model_path, cfg.exp_name, "trained_model")
        if cfg.loaded_iter == -1:
            loaded_iter = searchForMaxIteration(pretrain_model_dir)
        else:
            loaded_iter = cfg.loaded_iter
        if cfg.model.nsg.use_deformation_model: 
            gaussians.load_state_dict(pretrain_model_dir, loaded_iter)
        else:
            ckpt_path = os.path.join(pretrain_model_dir, f'iteration_{loaded_iter}.pth')
            state_dict = torch.load(ckpt_path)
            start_iter = state_dict['iter']
            # print(f'Loading model from {ckpt_path}')
            # breakpoint()
            # gaussians.load_state_dict(state_dict)
    gaussians.training_setup()
    print(f'Starting from {start_iter}')
    save_cfg(cfg, cfg.model_path, epoch=start_iter)

    gaussians_renderer = StreetGaussianRenderer()
    # cfg.model.nsg.use_deformation_model = False
    # scene_reconstruction(gaussians, scene, gaussians_renderer, "coarse", tb_writer)
    # cfg.model.nsg.use_deformation_model = True
    scene_reconstruction(gaussians, scene, gaussians_renderer, "fine", tb_writer)


    


def prepare_output_and_logger():
    
    # if cfg.model_path == '':
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str = os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #     cfg.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(cfg.model_path))
    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    os.makedirs(cfg.record_dir, exist_ok=True)
    if not cfg.resume:
        os.system('rm -rf {}/*'.format(cfg.record_dir))
        os.system('rm -rf {}/*'.format(cfg.trained_model_dir))

    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        viewer_arg = dict()
        viewer_arg['sh_degree'] = cfg.model.gaussian.sh_degree
        viewer_arg['white_background'] = cfg.data.white_background
        viewer_arg['source_path'] = cfg.source_path
        viewer_arg['model_path']= cfg.model_path
        cfg_log_f.write(str(Namespace(**viewer_arg)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, scalar_stats, tensor_stats, testing_iterations, scene: Scene, renderer: StreetGaussianRenderer):
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except:
            print('Failed to write to tensorboard')
            
            
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test/test_view', 'cameras' : scene.getTestCameras()},
                              {'name': 'test/train_view', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians, use_hexplane=cfg.model.nsg.use_deformation_model)["rgb"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if hasattr(viewpoint, 'original_mask'):
                        mask = viewpoint.original_mask.cuda().bool()
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                
                log_file_path = os.path.join(cfg.record_dir, 'metrics_log.txt')

                # Open the log file in append mode
                with open(log_file_path, 'a') as log_file:
                    # Write the metrics to the file
                    log_file.write(config['name'] + f" Iteration: {iteration}, L1 Loss: {l1_test}, PSNR: {psnr_test}\n")

        if tb_writer:
            # tb_writer.add_histogram("test/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('test/points_total', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Optimizing " + cfg.model_path)

    # Initialize system state (RNG)
    safe_state(cfg.train.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)
    training()

    # All done
    print("\nTraining complete.")