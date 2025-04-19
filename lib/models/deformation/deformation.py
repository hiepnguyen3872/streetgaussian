import functools
import math
import os
import time
from tkinter import W
from lib.config import cfg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from lib.utils.graphics_utils import batch_quaternion_multiply
from lib.models.deformation.hexplane import HexPlaneField
from lib.models.deformation.grid import DenseGrid
# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, max_embeddings=150, skips=[]):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.deform_W = W + cfg.model.nsg.embedding_feats_shape
        self.embedding_feats_shape = cfg.model.nsg.embedding_feats_shape
        # self.deform_W = cfg.model.nsg.embedding_feats_shape + cfg.model.deformable.temporal_embedding_dim
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = cfg.model.deformable.no_grid
        self.grid = HexPlaneField(cfg.model.deformable.bounds, cfg.model.kplanes_config, cfg.model.deformable.multires)
        # cfg.model.deformable.empty_voxel=True
        if cfg.model.deformable.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if cfg.model.deformable.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.deform_W,self.deform_W),nn.ReLU(),nn.Linear(self.deform_W, 1))
        
        # self.temporal_embedding_dim = cfg.model.deformable.temporal_embedding_dim
        # self.weight_time = torch.nn.Parameter(torch.normal(0., 0.01/np.sqrt(self.temporal_embedding_dim),size=(max_embeddings, self.temporal_embedding_dim)))
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if cfg.model.deformable.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        # if not cfg.model.deformable.no_dx :
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.deform_W,self.deform_W),nn.ReLU(),nn.Linear(self.deform_W, 3))
        # self.pos_vel = nn.Sequential(nn.ReLU(),nn.Linear(self.embedding_feats_shape,self.embedding_feats_shape),nn.ReLU(),nn.Linear(self.embedding_feats_shape, 3))
        # self.pos_vel = nn.Sequential(nn.ReLU(),nn.Linear(self.deform_W,self.deform_W),nn.ReLU(),nn.Linear(self.deform_W, 3))
        # self.pos_accelerator = nn.Sequential(nn.ReLU(),nn.Linear(self.deform_W,self.deform_W),nn.ReLU(),nn.Linear(self.deform_W, 3))
        self.pos_lifespan = nn.Sequential(nn.ReLU(),nn.Linear(self.embedding_feats_shape,self.embedding_feats_shape),nn.ReLU(),nn.Linear(self.embedding_feats_shape, 2))
        # if not cfg.model.deformable.no_ds :
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.deform_W,self.deform_W),nn.ReLU(),nn.Linear(self.deform_W, 3))
        # if not cfg.model.deformable.no_dr :
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.deform_W,self.deform_W),nn.ReLU(),nn.Linear(self.deform_W, 4))
        # if not cfg.model.deformable.no_do :
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.deform_W,self.deform_W),nn.ReLU(),nn.Linear(self.deform_W, 1))
        # self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))
        # if not cfg.model.deformable.no_dshs :
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.deform_W,self.deform_W),nn.ReLU(),nn.Linear(self.deform_W, 4*3))
        if cfg.model.deformable.feat_head:
            semantic_feature_dim = 64
            feature_mlp_layer_width = 64
            feature_embedding_dim = 3
            self.dino_head = nn.Sequential(
                nn.Linear(semantic_feature_dim, feature_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(feature_mlp_layer_width, feature_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(feature_mlp_layer_width, feature_embedding_dim),
            )

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            # 这里是 hexplane的forward 得到 feature [N, 128]
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 
        hidden = self.feature_out(hidden)   # [N,64]
        # time_emb = self.get_temporal_embed(time_emb[:,:1], current_num_embeddings=150)
        # if len(time_emb.shape) == 1:
        #     time_emb = time_emb.unsqueeze(0)  
        # hidden = torch.cat((hidden, time_emb), dim=1) 
 
        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None, embedding_feats = None, raw_time = None, is_current_time = True):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb, embedding_feats, raw_time, is_current_time)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb, embedding_feats, raw_time, is_current_time):
        # TODO: 这个hidden 应该考虑前后t
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        if cfg.model.nsg.embedding_feats_shape > 0: 
            hidden = torch.cat((hidden, embedding_feats), dim=1)
        # TODO: 这里需要重新考虑，如何得到静态的mask
        if cfg.model.deformable.static_mlp:
            mask = self.static_mlp(hidden)
        elif cfg.model.deformable.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1) # [N, 1]
        if cfg.model.deformable.no_dx:
            pts = rays_pts_emb[:,:3]
            dx = None
        else:
            dx = self.pos_deform(hidden) # [N, 3]
            if not is_current_time: 
                dx = dx.detach()
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx
        # else:
        #     dv = self.pos_vel(hidden) # [N, 3]
        #     da = self.pos_accelerator(hidden)
        #     if not is_current_time: 
        #         dv = dv.detach()
        #     # pts = torch.zeros_like(rays_pts_emb[:,:3])
        #     # pts = rays_pts_emb[:,:3]
        #     # pts[:, :2] = pts[:, :2]*mask + dx[:, :2]
        #     pts = torch.zeros_like(rays_pts_emb[:,:3])
        #     # ptx = rays_pts_emb[:,:3]*mask + dx*raw_time + (1/2)*da*raw_time*raw_time
        #     nor_time_emb = time_emb - (-1)
        #     dx = dv*nor_time_emb + (1/2)*da*nor_time_emb*nor_time_emb
        #     pts = rays_pts_emb[:,:3]*mask + dx

        if cfg.model.deformable.no_ds :
            
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            if not is_current_time: 
                ds = ds.detach()

            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            
        if cfg.model.deformable.no_dr :
            rotations = rotations_emb[:,:4]
            dr = None
        else:
            dr = self.rotations_deform(hidden)
            if not is_current_time: 
                dr = dr.detach()

            rotations = torch.zeros_like(rotations_emb[:,:4])
            if cfg.model.deformable.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        if cfg.model.deformable.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
            if not is_current_time: 
                do = do.detach()
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        if cfg.model.deformable.no_dshs:
            shs = shs_emb
            dshs = shs_emb
        else:
            # dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],4,3])
            if not is_current_time: 
                dshs = dshs.detach()
            shs = torch.zeros_like(shs_emb)
            shs = shs_emb*mask.unsqueeze(-1) + dshs

        feat = None
        if cfg.model.deformable.feat_head:
            feat = self.dino_head(hidden)
        
        # lifespan
        mu_t, log_sigma_t = self.pos_lifespan(embedding_feats).split(1, dim=-1)
        sigma_t = torch.exp(log_sigma_t)
        life_score = torch.exp(-((time_emb - mu_t)**2) / (2 * sigma_t**2))
        opacity *= life_score

        return pts, scales, rotations, opacity, shs, dx, feat, dshs, dr, mu_t, sigma_t
        # return pts, scales, rotations, opacity, shs, dx, feat, dshs, dr, dv, da
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
    
        
    def get_temporal_embed(self, t, current_num_embeddings, align_corners=True):
            emb_resized = F.interpolate(self.weight_time[None,None,...], 
                                    size=(current_num_embeddings, self.temporal_embedding_dim), 
                                    mode='bilinear', align_corners=True)
            N, _ = t.shape
            t = t[0,0]

            fdim = self.temporal_embedding_dim
            grid = torch.cat([torch.arange(fdim).cuda().unsqueeze(-1)/(fdim-1), torch.ones(fdim,1).cuda() * t, ], dim=-1)[None,None,...]
            grid = (grid - 0.5) * 2

            emb = F.grid_sample(emb_resized, grid, align_corners=align_corners, mode='bilinear', padding_mode='reflection')
            emb = emb.repeat(1,1,N,1).squeeze()

            return emb
        
class deform_network(nn.Module):
    def __init__(self) :
        super(deform_network, self).__init__()
        net_width = cfg.model.deformable.net_width
        timebase_pe = cfg.model.deformable.timebase_pe
        defor_depth= cfg.model.deformable.defor_depth
        posbase_pe= cfg.model.deformable.posebase_pe
        scale_rotation_pe = cfg.model.deformable.scale_rotation_pe
        opacity_pe = cfg.model.deformable.opacity_pe
        timenet_width = cfg.model.deformable.timenet_width
        timenet_output = cfg.model.deformable.timenet_output
        grid_pe = cfg.model.deformable.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, embedding_feats = None, raw_time = None, is_current_time=True):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel, embedding_feats, raw_time, is_current_time)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, embedding_feats = None, raw_time = None, is_current_time=True):
        ## times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        
        ## time_emb = poc_fre(times_sel, self.time_poc)
        ## times_feature = self.timenet(time_emb)
        means3D, scales, rotations, opacity, shs, dx, feat, dshs, dr, mu_t, sigma_t = self.deformation_net( point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel,
                                                embedding_feats,
                                                raw_time,
                                                is_current_time) # [N, 1]
        
        return means3D, scales, rotations, opacity, shs, dx , feat, dshs, dr, mu_t, sigma_t
        # return means3D, scales, rotations, opacity, shs, dx , feat, dshs, dr, dv, da
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb