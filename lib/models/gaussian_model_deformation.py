import torch
import numpy as np
import os
from lib.config import cfg
from lib.utils.graphics_utils import BasicPointCloud
from lib.datasets.base_readers import fetchPly
from lib.models.gaussian_model import GaussianModel
from lib.utils.camera_utils import Camera, make_rasterizer
from lib.models.deformation.deformation import deform_network
from lib.utils.general_utils import get_expon_lr_func
import torch.nn as nn
from lib.utils.general_utils import inverse_sigmoid, get_expon_lr_func, quaternion_to_matrix
from lib.models.deformation.regulation import compute_plane_smoothness
from lib.utils.loss_utils import get_dynamic_mask
class GaussianModelDeformable(GaussianModel):
    def __init__(
        self, 
        model_name='background', 
        scene_center=np.array([0, 0, 0]),
        scene_radius=20,
        sphere_center=np.array([0, 0, 0]),
        sphere_radius=20,
        stage='fine'
    ):
        self.scene_center = torch.from_numpy(scene_center).float().cuda()
        self.scene_radius = torch.tensor([scene_radius]).float().cuda()
        self.sphere_center = torch.from_numpy(sphere_center).float().cuda()
        self.sphere_radius = torch.tensor([sphere_radius]).float().cuda()
        num_classes = cfg.data.num_classes if cfg.data.get('use_semantic', False) else 0
        self.background_mask = None
        super().__init__(model_name=model_name, num_classes=num_classes, stage=stage)
        self._deformation = deform_network()
        self._deformation_table = torch.empty(0)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float): 
        print('Create background model')
        
        # pointcloud_path_sky =  os.path.join(cfg.model_path, 'input_ply', 'points3D_sky.ply')
        # include_sky = cfg.model.nsg.get('include_sky', False)
        # if os.path.exists(pointcloud_path_sky) and not include_sky:
        #     pcd_sky = fetchPly(pointcloud_path_sky)
        #     pointcloud_xyz = np.concatenate((pcd.points, pcd_sky.points), axis=0)
        #     pointcloud_rgb = np.concatenate((pcd.colors, pcd_sky.colors), axis=0)
        #     pointcloud_normal = np.zeros_like(pointcloud_xyz)
        #     pcd = BasicPointCloud(pointcloud_xyz, pointcloud_rgb, pointcloud_normal)
        # return super().create_from_pcd(pcd, spatial_lr_scale)
        super().create_from_pcd(pcd, spatial_lr_scale)
        self._deformation = self._deformation.to("cuda") 
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
    
    def save_deformation(self, path, iteration):
        torch.save(self._deformation.state_dict(),os.path.join(path, f"iteration_{iteration}_deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, f"iteration_{iteration}_deformation_table.pth"))
        torch.save(self._deformation_accum,os.path.join(path, f"iteration_{iteration}_deformation_accum.pth"))

    def set_background_mask(self, camera: Camera):
        pass
    
    @property
    def get_scaling(self):
        scaling = super().get_scaling
        return scaling if self.background_mask is None else scaling[self.background_mask]

    @property
    def get_rotation(self):
        rotation = super().get_rotation
        return rotation if self.background_mask is None else rotation[self.background_mask]

    @property
    def get_xyz(self):
        xyz = super().get_xyz
        return xyz if self.background_mask is None else xyz[self.background_mask]        
    
    @property
    def get_features(self):
        features = super().get_features
        return features if self.background_mask is None else features[self.background_mask]        
    
    @property
    def get_opacity(self):
        opacity = super().get_opacity
        return opacity if self.background_mask is None else opacity[self.background_mask]
    
    @property
    def get_semantic(self):
        semantic = super().get_semantic
        return semantic if self.background_mask is None else semantic[self.background_mask]
    
    @property
    def get_embedding_feats(self):
        embedding_feats = super().get_embedding_feats
        return embedding_feats if self.background_mask is None else embedding_feats[self.background_mask]
    
    @property
    def get_aabb(self):
        return self._deformation.get_aabb
    
    def training_setup(self):
        args = cfg.optim
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.active_sh_degree = 0
        l = [
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': args.deformable.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': args.deformable.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self._xyz], 'lr': args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': args.rotation_lr, "name": "rotation"},
            {'params': [self._semantic], 'lr': args.semantic_lr, "name": "semantic"},
            {'params': [self._embedding_feats], 'lr': args.embedding_feats_lr, "name": "embedding_feats"},
        ]
        
        self.percent_dense = args.percent_dense
        self.percent_big_ws = args.percent_big_ws
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=args.position_lr_init * self.spatial_lr_scale,
            lr_final=args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=args.position_lr_delay_mult,
            max_steps=args.position_lr_max_steps
        )
        
        self.densify_and_prune_list = ['xyz, f_dc, f_rest, opacity, scaling, rotation, semantic', 'embedding_feats']
        self.scalar_dict = dict()
        self.tensor_dict = dict() 

    def densify_and_prune(self, max_grad, min_opacity, prune_big_points):
        max_grad = cfg.optim.get('densify_grad_threshold_bkgd', max_grad)
        if cfg.optim.get('densify_grad_abs_bkgd', False):
            grads = self.xyz_gradient_accum[:, 1:2] / self.denom
        else:
            grads = self.xyz_gradient_accum[:, 0:1] / self.denom
        grads[grads.isnan()] = 0.0
        self.scalar_dict.clear()
        self.tensor_dict.clear()    
        self.scalar_dict['points_total'] = self.get_xyz.shape[0]

        # Clone and Split
        extent = self.scene_radius
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Prune points below opacity
        dynamic_mask = get_dynamic_mask(self.get_semantic.argmax(dim=1))
        prune_mask = torch.zeros_like(self.get_opacity, dtype=torch.bool)
        prune_mask[dynamic_mask] = (self.get_opacity[dynamic_mask] < min_opacity)
        prune_mask[~dynamic_mask] = (self.get_opacity[~dynamic_mask] < 2*min_opacity)
        prune_mask = prune_mask.squeeze()
        # prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.scalar_dict['points_below_min_opacity'] = prune_mask.sum().item()

        # Prune big points in world space 
        if prune_big_points:
            dists = torch.linalg.norm(self.get_xyz - self.sphere_center, dim=1)            
            big_points_ws = torch.max(self.get_scaling, dim=1).values > extent * self.percent_big_ws
            big_points_ws[dists > 2 * self.sphere_radius] = False
            
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
            
            self.scalar_dict['points_big_ws'] = big_points_ws.sum().item()

        self.scalar_dict['points_pruned'] = prune_mask.sum().item()
        self.prune_points(prune_mask)
        
        # Reset 
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()

        return self.scalar_dict, self.tensor_dict

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask, 
            prune_list = ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', 'semantic', 'embedding_feats'])

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic = optimizable_tensors["semantic"]
        self._embedding_feats = optimizable_tensors["embedding_feats"]

        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    def cat_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        name_list = tensors_dict.keys()
        if 'new_deformation_table' in name_list: 
            optimizable_tensors['new_deformation_table'] = tensors_dict['new_deformation_table']
        for group in self.optimizer.param_groups:
            if group['name'] not in name_list:
                continue
            
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def densification_postfix(self, tensors_dict):
        optimizable_tensors = self.cat_optimizer(tensors_dict)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic = optimizable_tensors["semantic"]
        self._embedding_feats = optimizable_tensors["embedding_feats"]
        
        cat_points_num = self.get_xyz.shape[0] - self.xyz_gradient_accum.shape[0]
        self._deformation_table = torch.cat([self._deformation_table,optimizable_tensors['new_deformation_table']],-1)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros(cat_points_num, 2).cuda()], dim=0)
        self.denom = torch.cat([self.denom, torch.zeros(cat_points_num, 1).cuda()], dim=0)
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(cat_points_num).cuda()], dim=0)

        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
                
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_extent = torch.zeros((n_init_points), device="cuda")
        padded_extent[:grads.shape[0]] = scene_extent
        
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * padded_extent)

        self.scalar_dict['points_split'] = selected_pts_mask.sum().item()
        print(f'Number of points to split: {selected_pts_mask.sum()}')

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_matrix(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_semantic = self._semantic[selected_pts_mask].repeat(N, 1)
        new_embedding_feats = self._embedding_feats[selected_pts_mask].repeat(N, 1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)

        self.densification_postfix({
            "xyz": new_xyz, 
            "f_dc": new_features_dc, 
            "f_rest": new_features_rest, 
            "opacity": new_opacity, 
            "scaling" : new_scaling, 
            "rotation" : new_rotation,
            "semantic" : new_semantic,
            "embedding_feats": new_embedding_feats,
            "new_deformation_table": new_deformation_table
        })

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask, 
            prune_list = ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', 'semantic', 'embedding_feats'])

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic = optimizable_tensors["semantic"]
        self._embedding_feats = optimizable_tensors["embedding_feats"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]

    
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        self.scalar_dict['points_clone'] = selected_pts_mask.sum().item()
        print(f'Number of points to clone: {selected_pts_mask.sum()}')
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantic = self._semantic[selected_pts_mask]
        
        # class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        #                'traffic light', 'traffic sign', 'vegetation', 'terrain', 
        #                'sky', 'person', 'rider', 'car', 'truck', 'bus', 
        #                'train', 'motorcycle', 'bicycle']
        
        # dynamic_classes = {'person', 'rider', 'car', 'truck', 'bus', 
        #                    'train', 'motorcycle', 'bicycle'}
        # dynamic_class_indices = [class_names.index(cls) for cls in dynamic_classes]
        # dynamic_class_indices_tensor = torch.tensor(dynamic_class_indices, device=new_semantic.device)
        # new_semantic.index_add_(1, dynamic_class_indices_tensor, torch.full_like(new_semantic[:, dynamic_class_indices_tensor], 300))
        
        
        new_embedding_feats = self._embedding_feats[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]

        self.densification_postfix({
            "xyz": new_xyz, 
            "f_dc": new_features_dc, 
            "f_rest": new_features_rest, 
            "opacity": new_opacity, 
            "scaling" : new_scaling, 
            "rotation" : new_rotation,
            "semantic" : new_semantic,
            "embedding_feats": new_embedding_feats,
            "new_deformation_table": new_deformation_table,
        })
    
    def add_point_by_mask(self, selected_pts_mask, perturb=0):
        selected_xyz = self._xyz[selected_pts_mask] 
        new_xyz, mask = self.get_displayment(selected_xyz, self.get_xyz.detach(),perturb)
        # displacements = torch.randn(selected_xyz.shape[0], 3).to(self._xyz) * perturb

        # new_xyz = selected_xyz + displacements
        # - 0.001 * self._xyz.grad[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask][mask]
        new_features_rest = self._features_rest[selected_pts_mask][mask]
        new_opacities = self._opacity[selected_pts_mask][mask]
        
        new_scaling = self._scaling[selected_pts_mask][mask]
        new_rotation = self._rotation[selected_pts_mask][mask]
        new_deformation_table = self._deformation_table[selected_pts_mask][mask]

        self.densification_postfix({
            "xyz": new_xyz, 
            "f_dc": new_features_dc, 
            "f_rest": new_features_rest, 
            "opacity": new_opacities, 
            "scaling" : new_scaling, 
            "rotation" : new_rotation,
            "new_deformation_table": new_deformation_table,
        })

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table)
        return selected_xyz, new_xyz
    
    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def _l1_regulation(self):
                # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()
    @torch.no_grad()
    def update_deformation_table(self,threshold):
        # print("origin deformation point nums:",self._deformation_table.sum())
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values/100,threshold)

    
    def load_state_dict(self, state_dict, trained_model_dir, loaded_iter): 
        # if cfg.mode == 'train': 
        #     self._xyz = nn.Parameter(torch.cat((self._xyz, state_dict['xyz']), dim=0))    
        #     self._features_dc = nn.Parameter(torch.cat((self._features_dc, state_dict['feature_dc']), dim=0))  
        #     self._features_rest = nn.Parameter(torch.cat((self._features_rest, state_dict['feature_rest']), dim=0))  
        #     self._scaling = nn.Parameter(torch.cat((self._scaling, state_dict['scaling']), dim=0))  
        #     self._rotation = nn.Parameter(torch.cat((self._rotation, state_dict['rotation']), dim=0))  
        #     self._opacity = nn.Parameter(torch.cat((self._opacity, state_dict['opacity']), dim=0))  
        #     self._semantic = nn.Parameter(torch.cat((self._semantic, state_dict['semantic']), dim=0))  
        # else: 
        if True:
            self._xyz = state_dict['xyz'].to('cuda')
            self._features_dc = state_dict['feature_dc'].to('cuda')
            self._features_rest = state_dict['feature_rest'].to('cuda')
            self._scaling = state_dict['scaling'].to('cuda')
            self._rotation = state_dict['rotation'].to('cuda')
            self._opacity = state_dict['opacity'].to('cuda')
            self._semantic = state_dict['semantic'].to('cuda')
        
        if "embedding_feats" in state_dict.keys():
            # self._embedding_feats = state_dict["embedding_feats"]
            # if cfg.mode == 'train': 
            #     self._embedding_feats = nn.Parameter(torch.cat((self._embedding_feats, state_dict['embedding_feats']), dim=0))
            # else: 
            if True:
                self._embedding_feats = state_dict["embedding_feats"].to('cuda')
        else: 
            self._embedding_feats = nn.Parameter(torch.zeros((self._semantic.shape[0], self.embedding_feats_shape), dtype=torch.float, device="cuda"))

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)

        
        if os.path.exists(os.path.join(trained_model_dir,f"iteration_{str(loaded_iter)}_deformation.pth")):
            print("Loading deformation network")
            weight_dict = torch.load(os.path.join(trained_model_dir,f"iteration_{str(loaded_iter)}_deformation.pth"),map_location="cuda")
            self._deformation.load_state_dict(weight_dict)
            self._deformation = self._deformation.to("cuda")
            self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
            self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(trained_model_dir,f"iteration_{str(loaded_iter)}_deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(trained_model_dir,f"iteration_{str(loaded_iter)}_deformation_table.pth"),map_location="cuda")
        if os.path.exists(os.path.join(trained_model_dir,f"iteration_{str(loaded_iter)}_deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(trained_model_dir,f"iteration_{str(loaded_iter)}_deformation_accum.pth"),map_location="cuda")
        if cfg.mode == 'train':
            self.training_setup()
            if 'spatial_lr_scale' in state_dict:
                self.spatial_lr_scale = state_dict['spatial_lr_scale'] 
            if 'denom' in state_dict:
                self.denom = state_dict['denom'] 
            if 'max_radii2D' in state_dict:
                self.max_radii2D = state_dict['max_radii2D'] 
            if 'xyz_gradient_accum' in state_dict:
                self.xyz_gradient_accum = state_dict['xyz_gradient_accum']
            if 'active_sh_degree' in state_dict:
                self.active_sh_degree = state_dict['active_sh_degree']
            if 'optimizer' in state_dict:
                self.optimizer.load_state_dict(state_dict['optimizer'])
