from .yacs import CfgNode as CN
import argparse
import os
import numpy as np

from lib.utils.cfg_utils import make_cfg

cfg = CN()

cfg.workspace = os.environ['PWD']
cfg.loaded_iter = -1
cfg.ip = '127.0.0.1'
cfg.port = 6009
cfg.data_device = 'cuda'
cfg.mode = 'train' 
cfg.task = 'hello' # task folder name
cfg.exp_name = 'test' # experiment folder name
cfg.gpus = [1] # list of gpus to use 
cfg.debug = False
cfg.resume = True # If set to True, resume training from the last checkpoint.

cfg.source_path = ''
cfg.model_path = ''
cfg.pretrain_model_path = '/media/ml4u/ExtremeSSD/datasets/log/TheHiep/StreetGaussian/semantic_warmup_only_deformation_dynamic_cls/coarse/waymo_exp'
cfg.record_dir = None
cfg.resolution = -1
cfg.resolution_scales = [1]

cfg.eval = CN()
cfg.eval.skip_train = False 
cfg.eval.skip_test = False 
cfg.eval.eval_train = False
cfg.eval.eval_test = True
cfg.eval.quiet = False

cfg.train = CN()
cfg.train.debug_from = -1
cfg.train.detect_anomaly = False
cfg.train.test_iterations = [7000, 30000]
cfg.train.save_iterations = [7000, 30000]
cfg.train.iterations = 30000
cfg.train.quiet = False
cfg.train.checkpoint_iterations = [30000]
cfg.train.start_checkpoint = None
cfg.train.importance_sampling = False

cfg.optim = CN()
# learning rate
cfg.optim.position_lr_init = 0.00016 # position_lr_init_{bkgd, obj ...}, similar to the following
cfg.optim.position_lr_final = 0.0000016
cfg.optim.position_lr_delay_mult = 0.01
cfg.optim.position_lr_max_steps = 30000
cfg.optim.feature_lr = 0.0025
cfg.optim.opacity_lr = 0.05
cfg.optim.scaling_lr = 0.005
cfg.optim.rotation_lr = 0.001
# densification and pruning
cfg.optim.percent_dense = 0.01 
cfg.optim.densification_interval = 100
cfg.optim.opacity_reset_interval = 3000
cfg.optim.densify_from_iter = 500
cfg.optim.densify_until_iter = 15000
cfg.optim.densify_grad_threshold = 0.0002 # densify_grad_threshold_{bkgd, obj ...}
cfg.optim.densify_grad_abs_bkgd = False # densification strategy from AbsGS
cfg.optim.densify_grad_abs_obj = False 
cfg.optim.max_screen_size = 20
cfg.optim.min_opacity = 0.005
cfg.optim.percent_big_ws = 0.1
# loss weight
cfg.optim.lambda_l1 = 1.
cfg.optim.lambda_dssim = 0.2
cfg.optim.lambda_sky = 0.
cfg.optim.lambda_sky_scale = []
cfg.optim.lambda_semantic = 0.
cfg.optim.lambda_reg = 0.
cfg.optim.lambda_depth_lidar = 0.
cfg.optim.lambda_depth_mono = 0.
cfg.optim.lambda_normal_mono = 0.
cfg.optim.lambda_color_correction = 0.
cfg.optim.lambda_pose_correction = 0.
cfg.optim.lambda_scale_flatten = 0.
cfg.optim.lambda_opacity_sparse = 0.


cfg.model = CN()
cfg.model.gaussian = CN()
cfg.model.gaussian.sh_degree = 3
cfg.model.gaussian.fourier_dim = 1 # fourier spherical harmonics dimension
cfg.model.gaussian.fourier_scale = 1.
cfg.model.gaussian.flip_prob = 0. # symmetry prior for rigid objects, flip gaussians with this probability during training
cfg.model.gaussian.semantic_mode = 'logits'

cfg.model.nsg = CN()
cfg.model.nsg.include_bkgd = True # include background
cfg.model.nsg.include_obj = True # include object
cfg.model.nsg.include_sky = False # include sky cubemap
cfg.model.nsg.include_pvg = False
cfg.model.nsg.opt_track = True # tracklets optimization
cfg.model.nsg.train_key_frame = False
cfg.model.nsg.embedding_feats_shape = 0
cfg.model.mask_2d_dynamic_object = False
cfg.model.sky = CN()
cfg.model.sky.resolution = 1024
cfg.model.sky.white_background = True
cfg.model.nsg.use_deformation_model = False

#### Note: We have not fully tested this.
cfg.model.use_color_correction = False # If set to True, learn transformation matrixs for appearance embedding
cfg.model.color_correction = CN() 
cfg.model.color_correction.mode = 'image' # If set to 'image', learn separate embedding for each image. If set to 'sensor', learn a single embedding for all images captured by one camera senosor. 
cfg.model.color_correction.use_mlp = False # If set to True, regress embedding from extrinsic by a mlp. Otherwise, define the embedding explicitly.
cfg.model.color_correction.use_sky = False # If set to True, using spparate embedding for background and sky
# Alternative choice from GOF: https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scene/appearance_network.py

cfg.model.use_pose_correction = False # If set to True, use pose correction for camera poses. 
cfg.model.pose_correction = CN()
cfg.model.pose_correction.mode = 'image' # If set to 'image', learn separate correction matrix for each image. If set to 'frame', learn a single correction matrix for all images corresponding to the same frame timestamp. 
####

cfg.data = CN()
cfg.data.generate_pseudo_view = False
cfg.data.white_background = False # If set to True, use white background. Should be False when using sky cubemap.
cfg.data.use_colmap_pose = False # If set to True, use colmap to recalibrate camera poses as input (rigid bundle adjustment now).
cfg.data.filter_colmap = False # If set to True, filter out SfM points by camera poses.
cfg.data.box_scale = 1.0 # Scale the bounding box by this factor.
cfg.data.split_test = -1 
cfg.data.shuffle = True
cfg.data.eval = True
cfg.data.type = 'Colmap'
cfg.data.images = 'images'
cfg.data.use_semantic = False
cfg.data.use_pseudo_depth = False
cfg.data.use_mono_depth = False
cfg.data.use_mono_normal = False
cfg.data.use_colmap = True
cfg.data.mask_lidar_dynamic_object = True
# data.load_pcd_from: Load the initialization point cloud from a previous experiment without generation.
# data.extent: radius of the scene, we recommend 10 - 20 meters.
# data.sphere_scale: Scale the sphere radius by this factor.
# data.regenerate_pcd: Regenerate the initialization point cloud.

cfg.render = CN()
cfg.render.convert_SHs_python = False
cfg.render.compute_cov3D_python = False
cfg.render.debug = False
cfg.render.scaling_modifier = 1.0
cfg.render.fps = 24
cfg.render.render_normal = False
cfg.render.save_video = True
cfg.render.save_image = True
cfg.render.coord = 'world' # ['world', 'vehicle']
cfg.render.concat_cameras = []
cfg.viewer = CN()
cfg.viewer.frame_id = 0 # Select the frame_id (start from 0) to save for viewer

cfg.model.deformable = CN()
cfg.model.deformable.smooth_render_dx_loss = False
cfg.model.deformable.viz_dx = False
cfg.model.deformable.use_adjoin_regular = False
cfg.model.deformable.temporal_embedding_dim = 0
cfg.model.deformable.net_width = 64
cfg.model.deformable.timebase_pe = 4
cfg.model.deformable.defor_depth = 1
cfg.model.deformable.posebase_pe = 10
cfg.model.deformable.scale_rotation_pe = 2
cfg.model.deformable.opacity_pe = 2
cfg.model.deformable.timenet_width = 64
cfg.model.deformable.timenet_output = 32
cfg.model.deformable.bounds = 1.6
cfg.model.deformable.plane_tv_weight = 0.0001
cfg.model.deformable.time_smoothness_weight = 0.01
cfg.model.deformable.l1_time_planes = 0.0001
cfg.model.deformable.multires = [1, 2, 4, 8]
cfg.model.deformable.no_dx=False
cfg.model.deformable.no_grid=False
cfg.model.deformable.no_ds=True 
cfg.model.deformable.no_dr=True
cfg.model.deformable.no_do=True
cfg.model.deformable.no_dshs=False
cfg.model.deformable.feat_head=True
cfg.model.deformable.empty_voxel=False
cfg.model.deformable.grid_pe=0
cfg.model.deformable.static_mlp=False
cfg.model.deformable.apply_rotation=False
cfg.model.kplanes_config = CN()
cfg.model.kplanes_configgrid_dimensions= 2
cfg.model.kplanes_config.input_coordinate_dim= 4
cfg.model.kplanes_config.output_coordinate_dim= 32
cfg.model.kplanes_config.resolution= [64, 64, 64, 25]

cfg.optim.deformable = CN()
cfg.optim.deformable.vis_step = 2000
cfg.optim.deformable.batch_size=1

cfg.optim.deformable.iterations = 50_000 # 30_000
cfg.optim.deformable.coarse_iterations = 500
cfg.optim.deformable.position_lr_init = 0.00016
cfg.optim.deformable.position_lr_final = 0.0000016

cfg.optim.deformable.position_lr_delay_mult = 0.01
cfg.optim.deformable.position_lr_max_steps = 30_00
cfg.optim.deformable.deformation_lr_init = 0.000016
cfg.optim.deformable.deformation_lr_final = 0.0000016
cfg.optim.deformable.deformation_lr_delay_mult = 0.01
cfg.optim.deformable.grid_lr_init = 0.00016
cfg.optim.deformable.grid_lr_final = 0.00001
cfg.optim.deformable.feature_lr = 0.0025
cfg.optim.deformable.opacity_lr = 0.05
cfg.optim.deformable.scaling_lr = 0.005
cfg.optim.deformable.rotation_lr = 0.001
cfg.optim.deformable.percent_dense = 0.01
cfg.optim.deformable.lambda_dssim = 0.2
cfg.optim.deformable.lambda_depth = 0.5
cfg.optim.deformable.densification_interval = 100   # 100
cfg.optim.deformable.opacity_reset_interval = 3000
cfg.optim.deformable.pruning_interval = 100
cfg.optim.deformable.pruning_from_iter = 500
cfg.optim.deformable.densify_until_iter = 25_000
# cfg.optim.deformable.densify_grad_threshold = 0.0002
cfg.optim.deformable.densify_grad_threshold_coarse = 0.0002
cfg.optim.deformable.densify_grad_threshold_fine_init = 0.0002
cfg.optim.deformable.densify_grad_threshold_after = 0.000
# cfg.optim.deformable.min_opacity_threshold = 0.005
cfg.optim.deformable.opacity_threshold_coarse = 0.005
cfg.optim.deformable.opacity_threshold_fine_init = 0.005
cfg.optim.deformable.opacity_threshold_fine_after = 0.00
cfg.optim.deformable.random_background = False
# for waymo
cfg.optim.deformable.max_points = 500_000
cfg.optim.deformable.prune_from_iter = 500
cfg.optim.deformable.prune_interval = 100

cfg.optim.deformable.scale_ratio = 1.0 #   global-scale = local-norm-scale * voxel_size * scale_ratio
# feat
cfg.optim.deformable.include_feature = True
cfg.optim.deformable.language_feature_lr = 0.0025 # TODO: update
cfg.optim.deformable.feat_dim = 8 #12  #  recomplie-cuda   SET DISTUTILS_USE_SDK=1
cfg.optim.deformable.feat_conv_lr = 0.0001
cfg.optim.deformable.lambda_feat = 0.001
cfg.optim.deformable.dx_reg = False
cfg.optim.deformable.lambda_dx = 0.001
cfg.optim.deformable.lambda_dshs = 0.001
# TODO: don't use, clean
cfg.optim.deformable.use_bg_gs = True
cfg.optim.deformable.use_bg_model = False
cfg.optim.deformable.bg_aabb_scale = 20.0 #2
cfg.optim.deformable.bg_gs_num = 5000
cfg.optim.deformable.bg_percent_dense = 0.01 #0.01
cfg.optim.deformable.bg_model_type = 'gs' #'mlp'
cfg.optim.deformable.mlp_width = 256
cfg.optim.deformable.bg_grid_res = 10 #　aabb/grid_res = grid_size
cfg.optim.deformable.bg_model_lr = 0.0025
cfg.optim.deformable.custom_xyz_scheduler = False
        
# deprecated
cfg.optim.deformable.densify_from_iter = 500   # 调整至与position_lr_after_iter 一致  # 500
cfg.optim.deformable.position_lr_after_iter = 500
cfg.optim.deformable.scale_ratio_threshold = 5.0 
cfg.optim.deformable.hard_alpha_composite = True
cfg.optim.deformable.alpha_mask_threshold = 0.8

parser = argparse.ArgumentParser()
# parser.add_argument("--config", default="configs/default.yaml", type=str)
parser.add_argument("--config", default="/home/ml4u/BKTeam/TheHiep/street_gaussians/configs/experiments_waymo/semantic_and_pseudo_depth/waymo_val_150.yaml", type=str)
parser.add_argument("--mode", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
cfg = make_cfg(cfg, args)