import torch
from lib.utils.sh_utils import eval_sh
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.camera_utils import Camera, make_rasterizer
from lib.config import cfg
from lib.utils.positional_embedding import positionalencoding1d
class StreetGaussianRenderer():
    def __init__(
        self,         
    ):
        self.cfg = cfg.render
              
    def render_all(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):
        
        # render all
        render_composition = self.render(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)

        # render background
        render_background = self.render_background(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        
        # render object
        render_object = self.render_object(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        
        result = render_composition
        result['rgb_background'] = render_background['rgb']
        result['acc_background'] = render_background['acc']
        result['rgb_object'] = render_object['rgb']
        result['acc_object'] = render_object['acc']
        
        # result['bboxes'], result['bboxes_input'] = pc.get_bbox_corner(viewpoint_camera)
    
        return result
    
    def render_object(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):        
        pc.set_visibility(include_list=pc.obj_list)
        pc.parse_camera(viewpoint_camera)
        
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, white_background=True)

        return result
    
    def render_background(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):
        pc.set_visibility(include_list=['background'])
        pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, white_background=True)

        return result
    
    def render_sky(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):  
        pc.set_visibility(include_list=['sky'])
        pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        return result
    
    def render(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        exclude_list = [],
        return_decomposition=False,
        return_dx=False,
        render_feat=False, 
        use_hexplane=False,
        other=None,
        time_shift=None,
        only_backprop_hexplane=False
    ):   
        include_list = list(set(pc.model_name_id.keys()) - set(exclude_list))
                    
        # Step1: render foreground
        pc.set_visibility(include_list)
        pc.parse_camera(viewpoint_camera)
        
        result = self.render_kernel(viewpoint_camera, 
                                    pc, 
                                    convert_SHs_python, 
                                    compute_cov3D_python, 
                                    scaling_modifier, 
                                    override_color, 
                                    return_decomposition=return_decomposition,
                                    return_dx=return_dx,
                                    render_feat=render_feat,
                                    use_hexplane=use_hexplane,
                                    other=other,
                                    time_shift=time_shift,
                                    only_backprop_hexplane=only_backprop_hexplane)

        # Step2: render sky
        if pc.include_sky:
            sky_color = pc.sky_cubemap(viewpoint_camera, result['acc'].detach())

            result['rgb'] = result['rgb'] + sky_color * (1 - result['acc'])

        if pc.use_color_correction:
            result['rgb'] = pc.color_correction(viewpoint_camera, result['rgb'])

        if cfg.mode != 'train':
            result['rgb'] = torch.clamp(result['rgb'], 0., 1.)
        return result
    
            
    def render_kernel(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        white_background = cfg.data.white_background,
        return_decomposition=False,
        return_dx=False,
        render_feat=False,
        use_hexplane=False,
        other=None,
        time_shift=None,
        only_backprop_hexplane=False,
    ):
        if pc.num_gaussians == 0:
            if white_background:
                rendered_color = torch.ones(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            else:
                rendered_color = torch.zeros(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            
            rendered_acc = torch.zeros(1, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            rendered_semantic = torch.zeros(0, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            
            return {
                "rgb": rendered_color,
                "acc": rendered_acc,
                "semantic": rendered_semantic,
            }

        # Set up rasterization configuration and make rasterizer
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_color).float().cuda()
        scaling_modifier = scaling_modifier or self.cfg.scaling_modifier
        rasterizer = make_rasterizer(viewpoint_camera, pc.max_sh_degree, bg_color, scaling_modifier)
        convert_SHs_python = convert_SHs_python or self.cfg.convert_SHs_python
        compute_cov3D_python = compute_cov3D_python or self.cfg.compute_cov3D_python

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        if cfg.mode == 'train':
            screenspace_points = torch.zeros((pc.num_gaussians, 3), requires_grad=True).float().cuda() + 0
            try:
                screenspace_points.retain_grad()
            except:
                pass
        else:
            screenspace_points = None 

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity
        # embedding_feats = pc.get_embedding_feats

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        # if cfg.mode == 'train': 
        #     if torch.rand(1).item() > 0.5:  # Check if random value is greater than 50%
        #         noise = torch.rand_like(time) * 0.3  # Generate noise in the range [0, 0.1]
        #         time += noise  # Add noise to the time feature
        
        # if cfg.model != 'train': 
        #     print('normalize time: ', time[0])
        #     print('raw time: ', viewpoint_camera.raw_time)
            
        if cfg.model.nsg.include_pvg: 
            if time_shift is not None:
                means3D = pc.background.get_xyz_SHM(viewpoint_camera.time-time_shift)
                means3D = means3D + pc.background.get_inst_velocity * time_shift
                marginal_t = pc.background.get_marginal_t(viewpoint_camera.time-time_shift)
            else:
                means3D = pc.background.get_xyz_SHM(viewpoint_camera.time)
                marginal_t = pc.background.get_marginal_t(viewpoint_camera.time)
            opacity = opacity * marginal_t

        if compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                try:
                    shs = pc.get_features
                except:
                    colors_precomp = pc.get_colors(viewpoint_camera.camera_center)
        else:
            colors_precomp = override_color
            
        # TODO: add more feature here
        feature_names = []
        feature_dims = []
        features = []
        
        if cfg.render.render_normal:
            normals = pc.get_normals(viewpoint_camera)
            feature_names.append('normals')
            feature_dims.append(normals.shape[-1])
            features.append(normals)
        
        if cfg.data.get('use_semantic', False):
            semantics = pc.get_semantic
            feature_names.append('semantic')
            feature_dims.append(semantics.shape[-1])
            features.append(semantics)
        
        if other is not None:
            feature_names.append('other')
            feature_dims.append(torch.cat(other, dim=1).shape[-1])
            features.append(torch.cat(other, dim=1).to(device='cuda'))

        if cfg.data.use_semantic:
            class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                        'traffic light', 'traffic sign', 'vegetation', 'terrain', 
                        'sky', 'person', 'rider', 'car', 'truck', 'bus', 
                        'train', 'motorcycle', 'bicycle']
            
            dynamic_classes = {'person', 'rider', 'car', 'truck', 'bus', 
                            'train', 'motorcycle', 'bicycle'}
            
            predicted_classes = torch.argmax(semantics, dim=1)  
            
            dynamic_class_indices = [class_names.index(cls) for cls in dynamic_classes]
            
            # Create dynamic mask
            dynamic_mask = torch.isin(predicted_classes, torch.tensor(dynamic_class_indices, device=semantics.device))
            print("number of dynamic gaussians: ", len(dynamic_mask[dynamic_mask]))
        else: 
            dynamic_mask = torch.ones(means3D.shape[0], dtype=torch.bool)

        # dynamic_mask = torch.ones_like(predicted_classes, dtype=torch.bool)  # Create a mask with all True
        selected_frames = cfg.data.selected_frames
        time_length = selected_frames[-1] - selected_frames[0]
        dx = None
        
        if cfg.model.nsg.include_pvg: 
            mask = marginal_t[:, 0] > 0.05
            masked_means3D = means3D[mask]
            masked_xyz_homo = torch.cat([masked_means3D, torch.ones_like(masked_means3D[:, :1])], dim=1)
            masked_depth = (masked_xyz_homo @ viewpoint_camera.world_view_transform[:, 2:3])
            depth_alpha = torch.zeros(means3D.shape[0], 2, dtype=torch.float32, device=means3D.device)
            depth_alpha[mask] = torch.cat([
                masked_depth,
                torch.ones_like(masked_depth)
            ], dim=1)
            # features = torch.cat([features, depth_alpha], dim=1)
        dx = torch.zeros_like(means3D) 
        dshs = torch.zeros_like(shs)
        dr = None
        mu_t = None
        sigma_t = None
        if cfg.model.nsg.use_deformation_model and len(dynamic_mask[dynamic_mask]) > 0:
            embedding_feats = pc.get_embedding_feats
            # means3D, scales, rotations, opacity, shs, dx, feat, dshs, dr = pc.background._deformation(means3D.detach(), scales.detach(), rotations.detach(), opacity.detach(), shs.detach(), time)
            # dx = None
            # dshs = None
            # dr = None
            if use_hexplane:
                if only_backprop_hexplane: 
                    means3D[~dynamic_mask] = means3D[~dynamic_mask].detach()
                    scales[~dynamic_mask] = scales[~dynamic_mask].detach()
                    rotations[~dynamic_mask] = rotations[~dynamic_mask].detach()
                    opacity[~dynamic_mask] = opacity[~dynamic_mask].detach()
                    shs[~dynamic_mask] = shs[~dynamic_mask].detach()
                means3D[dynamic_mask], scales[dynamic_mask], rotations[dynamic_mask], opacity[dynamic_mask], shs[dynamic_mask], dx[dynamic_mask], feat, dshs[dynamic_mask], dr, mu_t, sigma_t = pc.background._deformation(means3D[dynamic_mask], scales[dynamic_mask], rotations[dynamic_mask], opacity[dynamic_mask], shs[dynamic_mask], time[dynamic_mask], embedding_feats[dynamic_mask], raw_time = viewpoint_camera.raw_time - selected_frames[0], is_current_time = True)
                # means3D[dynamic_mask], scales[dynamic_mask], rotations[dynamic_mask], opacity[dynamic_mask], shs[dynamic_mask], dx[dynamic_mask], feat, dshs[dynamic_mask], dr, dv, da = pc.background._deformation(means3D[dynamic_mask], scales[dynamic_mask], rotations[dynamic_mask], opacity[dynamic_mask], shs[dynamic_mask], time[dynamic_mask], embedding_feats[dynamic_mask], raw_time = viewpoint_camera.raw_time - selected_frames[0], is_current_time = True)
                # means3D[sdynamic_mask], scales[dynamic_mask], rotations[dynamic_mask], opacity[dynamic_mask], shs[dynamic_mask], dx, feat, dshs, dr = pc.background._deformation(means3D[dynamic_mask], scales[dynamic_mask], rotations[dynamic_mask], opacity[dynamic_mask], shs[dynamic_mask], time[dynamic_mask], embedding_feats[dynamic_mask], is_current_time = True)
                # Loss constraint prediction of hexplane must equal in two consecutive frame pairs
                # if cfg.model.deformable.use_adjoin_regular: 
                if False: 
                    dx_adjoin_diff = None
                    dshs_adjoin_diff = None
                    # if cfg.model.deformable.use_adjoin_regular:
                    if True:
                        raw_time = viewpoint_camera.raw_time
                        # for timelapse in range(1,3):
                        if True: 
                            timelapse = 1
                            if raw_time >= selected_frames[-1] - timelapse - 1:
                                nor_time_adjoin = (raw_time-timelapse-selected_frames[0])/time_length
                            else: 
                                nor_time_adjoin = (raw_time+timelapse-selected_frames[0])/time_length
                            nor_time_adjoin = torch.tensor(nor_time_adjoin).to(means3D.device).repeat(means3D.shape[0],1)
                            # means3D[dynamic_mask], scales[dynamic_mask], rotations[dynamic_mask], opacity[dynamic_mask], shs[dynamic_mask], dx, feat, dshs, dr = pc.background._deformation(means3D[dynamic_mask], scales[dynamic_mask], rotations[dynamic_mask], opacity[dynamic_mask], shs[dynamic_mask], nor_time_adjoin[dynamic_mask], embedding_feats[dynamic_mask], is_current_time = True)
                            _, _, _, _, _, dx_adjoin, feat_adjoin, dshs_adjoin, dr_adjoin = pc.background._deformation(means3D[dynamic_mask].detach(), scales[dynamic_mask].detach(), rotations[dynamic_mask].detach(), opacity[dynamic_mask].detach(), shs[dynamic_mask].detach(), nor_time_adjoin[dynamic_mask].detach(), embedding_feats[dynamic_mask].detach(),is_current_time = True)
                            if dx_adjoin_diff is None:
                                dx_adjoin_diff = dx[dynamic_mask] - dx_adjoin
                            else: 
                                dx_adjoin_diff += (dx[dynamic_mask] - dx_adjoin)
                            
                            if dshs_adjoin_diff is None:
                                dshs_adjoin_diff = dshs[dynamic_mask] - dshs_adjoin
                            else: 
                                dshs_adjoin_diff += (dshs_adjoin[dynamic_mask] - dshs_adjoin)
                        dx[dynamic_mask] = dx_adjoin_diff
                        dshs[dynamic_mask] = dshs_adjoin_diff
                
                if False: 
                    # if cfg.model.deformable.use_adjoin_regular:
                    da_dv_loss = None
                    smooth_dv_loss = None
                    smooth_da_loss = None
                    if True:
                        raw_time = viewpoint_camera.raw_time
                        # for timelapse in range(1,3):
                        if True: 
                            timelapse = 1
                            if raw_time >= selected_frames[-1] - timelapse - 1:
                                nor_time_adjoin = (raw_time-timelapse-selected_frames[0])/time_length
                            else: 
                                nor_time_adjoin = (raw_time+timelapse-selected_frames[0])/time_length
                            nor_time_adjoin = torch.tensor(nor_time_adjoin).to(means3D.device).repeat(means3D.shape[0],1)
                            # means3D[dynamic_mask], scales[dynamic_mask], rotations[dynamic_mask], opacity[dynamic_mask], shs[dynamic_mask], dx, feat, dshs, dr = pc.background._deformation(means3D[dynamic_mask], scales[dynamic_mask], rotations[dynamic_mask], opacity[dynamic_mask], shs[dynamic_mask], nor_time_adjoin[dynamic_mask], embedding_feats[dynamic_mask], is_current_time = True)
                            _, _, _, _, _, dx_adjoin, feat_adjoin, dshs_adjoin, dr_adjoin, dv_adjoin, da_adjoin = pc.background._deformation(means3D[dynamic_mask].detach(), scales[dynamic_mask].detach(), rotations[dynamic_mask].detach(), opacity[dynamic_mask].detach(), shs[dynamic_mask].detach(), nor_time_adjoin[dynamic_mask].detach(), embedding_feats[dynamic_mask].detach(),is_current_time = True)
                            estimated_accel = (dv_adjoin - dv) / (nor_time_adjoin[dynamic_mask] - time[dynamic_mask])
                            da_dv_loss =  torch.mean((da - estimated_accel) ** 2)
                            smooth_dv_loss = torch.mean((dv_adjoin - dv) ** 2)
                            smooth_da_loss =   torch.mean((da_adjoin - da) ** 2) + torch.mean(da ** 2)
                        
                                
                # means3D[dynamic_mask], scales[dynamic_mask], rotations[dynamic_mask], opacity[dynamic_mask], shs[dynamic_mask], dx, feat, dshs, dr = pc.background._deformation(means3D[dynamic_mask].detach(), scales[dynamic_mask].detach(), rotations[dynamic_mask].detach(), opacity[dynamic_mask].detach(), shs[dynamic_mask].detach(), time[dynamic_mask].detach(), embedding_feats[dynamic_mask],is_current_time = True)
                # means3D[~dynamic_mask] = means3D[~dynamic_mask].detach()
                # scales[~dynamic_mask] = scales[~dynamic_mask].detach()
                # rotations[~dynamic_mask] = rotations[~dynamic_mask].detach()
                # opacity[~dynamic_mask] = opacity[~dynamic_mask].detach()
                # shs[~dynamic_mask] = shs[~dynamic_mask].detach()
            # for raw_time in range(selected_frames[0], viewpoint_camera.raw_time+1): 
            #     nor_time = (raw_time-selected_frames[0])/time_length
            #     nor_time = torch.tensor(nor_time).to(means3D.device).repeat(means3D.shape[0],1)
            #     is_current_time = raw_time == viewpoint_camera.raw_time
            #     means3D[dynamic_mask], scales[dynamic_mask], rotations[dynamic_mask], opacity[dynamic_mask], shs[dynamic_mask], dx, feat, dshs, dr = pc.background._deformation(means3D[dynamic_mask], scales[dynamic_mask], rotations[dynamic_mask], opacity[dynamic_mask], shs[dynamic_mask], nor_time[dynamic_mask], is_current_time=is_current_time)
                # if nor_time.sum() == 0:
                #     dx = dx_
                #     dshs = dshs_
                #     dr = dr_
                # if nor_time == time:
                # else: 
                #     if dx_ is not None:
                #         dx += dx_.detach()
                #     if dshs_ is not None:
                #         dshs += dshs_.detach()
                #     if dr_ is not None:
                #         dr += dr_.detach()
                
                # if cfg.model.deformable.viz_dx:
                #     feature_names.append('render_dx')
                #     feature_dims.append(dx.shape[-1])
                #     features.append(dx)
        
        if len(features) > 0:
            features = torch.cat(features, dim=-1)
        else:
            features = None
                
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        # rendered_color, radii, rendered_depth, rendered_acc, rendered_feature, proj_means_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(
        #     means3D = means3D,
        #     means2D = means2D,
        #     opacities = opacity,
        #     shs = shs,
        #     colors_precomp = colors_precomp,
        #     scales = scales,
        #     rotations = rotations,
        #     cov3D_precomp = cov3D_precomp,
        #     semantics = features
        #     )  

        rendered_color, radii, rendered_depth, rendered_acc, rendered_feature = rasterizer(
            means3D = means3D,
            means2D = means2D,
            opacities = opacity,
            shs = shs,
            colors_precomp = colors_precomp,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            semantics = features
            )  
        
        if cfg.mode != 'train':
            rendered_color = torch.clamp(rendered_color, 0., 1.)
        
        # rendered_feature_dict = dict()
        # if rendered_feature.shape[0] > 0:
        #     rendered_feature_list = torch.split(rendered_feature, feature_dims, dim=0)
        #     for i, feature_name in enumerate(feature_names):
        #         rendered_feature_dict[feature_name] = rendered_feature_list[i]
        
        # if 'normals' in rendered_feature_dict:
        #     rendered_feature_dict['normals'] = torch.nn.functional.normalize(rendered_feature_dict['normals'], dim=0)
                
        # if 'semantic' in rendered_feature_dict:
        #     rendered_semantic = rendered_feature_dict['semantic']
        #     semantic_mode = cfg.model.gaussian.get('semantic_mode', 'logits')
        #     assert semantic_mode in ['logits', 'probabilities']
        #     if semantic_mode == 'logits': 
        #         pass # return raw semantic logits
        #     else:
        #         rendered_semantic = rendered_semantic / (torch.sum(rendered_semantic, dim=0, keepdim=True) + 1e-8) # normalize to probabilities
        #         rendered_semantic = torch.log(rendered_semantic + 1e-8) # change for cross entropy loss

        #     rendered_feature_dict['semantic'] = rendered_semantic
        
        rendered_feature_dict = self.get_feature(rendered_feature, feature_names, feature_dims)
        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        
        result = {
            "rgb": rendered_color,
            "acc": rendered_acc,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii, 
            "gaussian_dynamic_mask": dynamic_mask, 
            "mu_t": mu_t,
            "sigma_t": sigma_t,
            "means3D": means3D,
            # "da_dv_loss": da_dv_loss,
            # "smooth_dv_loss": smooth_dv_loss,
            # "smooth_da_loss": smooth_da_loss,
            # "proj_means_2D": proj_means_2_per_gs_pixel,  # Added key for weight_per_gs_pixel
            # "x_mu": x_mu  D,  # Added key for proj_means_2D
            # "conic_2D": conic_2D,            # Added key for conic_2D
            # "conic_2D_inv": conic_2D_inv,    # Added key for conic_2D_inv
            # "gs_per_pixel": gs_per_pixel,    # Added key for gs_per_pixel
            # "weight_per_gs_pixel": weight
        }
        
        result.update(rendered_feature_dict)
        
        
        if use_hexplane and screenspace_points is not None and dx is not None and cfg.model.mask_2d_dynamic_object:
        # if False:
            # dx_abs = torch.abs(dx) # [N,3]
            # max_values = torch.max(dx_abs, dim=1)[0] # [N]
            # thre = torch.mean(max_values)
            
            # dynamic_mask = max_values > thre
            # dynamic_points = np.sum(dynamic_mask).item()
            rendered_image_d, radii_d, depth_d, _, rendered_feature_d = rasterizer(
                means3D = means3D[dynamic_mask].detach(),
                means2D = means2D[dynamic_mask].detach(),
                shs = shs[dynamic_mask].detach() if shs is not None else None,
                colors_precomp = colors_precomp[dynamic_mask].detach() if colors_precomp is not None else None, # [N,3]
                opacities = opacity[dynamic_mask],
                scales = scales[dynamic_mask].detach(),
                rotations = rotations[dynamic_mask].detach(),
                cov3D_precomp = cov3D_precomp[dynamic_mask].detach() if cov3D_precomp is not None else None,
                semantics = features[dynamic_mask].detach(),)
            rendered_feature_dict_d = self.get_feature(rendered_feature_d, [feature_name + '_d' for feature_name in feature_names], feature_dims)
            
            # rendered_image_s, radii_s, depth_s, _, rendered_feature_s = rasterizer(
            #     means3D = means3D[~dynamic_mask],
            #     means2D = means2D[~dynamic_mask],
            #     shs = shs[~dynamic_mask] if shs is not None else None,
            #     colors_precomp = colors_precomp[~dynamic_mask] if colors_precomp is not None else None, # [N,3]
            #     opacities = opacity[~dynamic_mask],
            #     scales = scales[~dynamic_mask],
            #     rotations = rotations[~dynamic_mask],
            #     cov3D_precomp = cov3D_precomp[~dynamic_mask] if cov3D_precomp is not None else None,
            #     semantics = features[~dynamic_mask].detach(),)
            # rendered_feature_dict_s = self.get_feature(rendered_feature_s, [feature_name + '_s' for feature_name in feature_names], feature_dims)
            result.update({
                "rgb_d": rendered_image_d,
                "depth_d":depth_d,
                "visibility_filter_d" : radii_d > 0,
                # "rgb_s": rendered_image_s,
                # "depth_s":depth_s,
                # "visibility_filter_s" : radii_s > 0,
                })
            result.update(rendered_feature_dict_d)
            # result.update(rendered_feature_dict_s)
        
        if cfg.model.deformable.viz_dx:
            _, _, _, _, render_dx = rasterizer(
            means3D = means3D.detach(),
            means2D = means2D.detach() if means2D is not None else None,
            opacities = opacity.detach(),
            shs = shs.detach(),
            colors_precomp = colors_precomp.detach() if colors_precomp is not None else None,
            scales = scales.detach(),
            rotations = rotations.detach(),
            cov3D_precomp = cov3D_precomp.detach() if cov3D_precomp is not None else None,
            semantics = dx
            )  
            result.update({'render_dx': render_dx})
        if return_dx:
            result.update({"dx": dx})
            result.update({'dshs' : dshs})
            result.update({'dr' : dr})
        
        return result
    
    def get_feature(self,  rendered_feature, feature_names, feature_dims): 
        rendered_feature_dict = dict()
        if rendered_feature.shape[0] > 0:
            rendered_feature_list = torch.split(rendered_feature, feature_dims, dim=0)
            for i, feature_name in enumerate(feature_names):
                rendered_feature_dict[feature_name] = rendered_feature_list[i]
        
        if 'normals' in rendered_feature_dict:
            rendered_feature_dict['normals'] = torch.nn.functional.normalize(rendered_feature_dict['normals'], dim=0)
                
        if 'semantic' in rendered_feature_dict:
            rendered_semantic = rendered_feature_dict['semantic']
            semantic_mode = cfg.model.gaussian.get('semantic_mode', 'logits')
            assert semantic_mode in ['logits', 'probabilities']
            if semantic_mode == 'logits': 
                pass # return raw semantic logits
            else:
                rendered_semantic = rendered_semantic / (torch.sum(rendered_semantic, dim=0, keepdim=True) + 1e-8) # normalize to probabilities
                rendered_semantic = torch.log(rendered_semantic + 1e-8) # change for cross entropy loss

            rendered_feature_dict['semantic'] = rendered_semantic
        
        return rendered_feature_dict