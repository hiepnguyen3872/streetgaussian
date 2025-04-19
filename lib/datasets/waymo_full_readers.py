from lib.utils.waymo_utils import generate_dataparser_outputs
from lib.utils.graphics_utils import focal2fov, BasicPointCloud
from lib.utils.data_utils import get_val_frames
from lib.datasets.base_readers import CameraInfo, PseudoCameraInfo,SceneInfo, getNerfppNorm, fetchPly, get_PCA_Norm, get_Sphere_Norm
from lib.config import cfg
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
import cv2
import sys
import copy
import shutil
sys.path.append(os.getcwd())
# from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torch
import scipy.ndimage


def warp_I_t_to_I_tprime_with_mask(I_t, depth_t, T_t_to_tprime, K, sky_mask):
    """
    Warp image I_t from time t to time t' and generate a visible mask.
    
    Args:
        I_t: Input image at time t. Can be a PIL Image or a torch tensor of shape (C, H, W) or (B, C, H, W).
        depth_t: Depth map at time t. Can be a NumPy array or torch tensor of shape (H, W) or (1, H, W) or (B, 1, H, W).
        T_t_to_tprime: Transformation matrix from t to t'. Can be a NumPy array or torch tensor of shape (4, 4) or (B, 4, 4).
        K: Camera intrinsics. Can be a NumPy array or torch tensor of shape (3, 3) or (B, 3, 3).
    
    Returns:
        I_tprime_warped (tensor): Warped image at time t', shape (B, C, H, W).
        visible_mask (tensor): Binary mask indicating valid warped regions, shape (B, 1, H, W).
    """
    # Convert I_t from PIL Image to tensor if necessary.
    if isinstance(I_t, Image.Image):
        I_t = ToTensor()(I_t)  # shape becomes (C, H, W)
    
    # Ensure I_t is 4D: (B, C, H, W)
    if I_t.dim() == 3:
        I_t = I_t.unsqueeze(0)
    I_t = I_t.float()
    
    # Convert depth_t to tensor if it's a NumPy array.
    if isinstance(depth_t, np.ndarray):
        depth_t = torch.from_numpy(depth_t).float()
    # If depth_t is 2D (H, W), add channel and batch dims: becomes (1, 1, H, W)
    if depth_t.dim() == 2:
        depth_t = depth_t.unsqueeze(0).unsqueeze(0)
    elif depth_t.dim() == 3:
        depth_t = depth_t.unsqueeze(0)  # now (1, 1, H, W) if originally (1, H, W)
    
    # Convert T_t_to_tprime to tensor if needed.
    if isinstance(T_t_to_tprime, np.ndarray):
        T_t_to_tprime = torch.from_numpy(T_t_to_tprime).float()
    # If T_t_to_tprime is 2D (4, 4), add a batch dimension.
    if T_t_to_tprime.dim() == 2:
        T_t_to_tprime = T_t_to_tprime.unsqueeze(0)
    
    # Convert K to tensor if needed.
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()
    if K.dim() == 2:
        K = K.unsqueeze(0)
    
    # Use the device of I_t for all tensors.
    device = I_t.device
    depth_t = depth_t.to(device)
    T_t_to_tprime = T_t_to_tprime.to(device)
    K = K.to(device)
    sky_mask = sky_mask.to(device)
    
    # Get image dimensions (from I_t) and depth dimensions.
    B, C, H_img, W_img = I_t.shape
    _, _, H_depth, W_depth = depth_t.shape
    # Resample depth to match image resolution if needed.
    if (H_img != H_depth) or (W_img != W_depth):
        depth_t = F.interpolate(depth_t, size=(H_img, W_img), mode='bilinear', align_corners=True)
    
    # Create mesh grid for pixel coordinates (B, 3, H_img * W_img)
    y, x = torch.meshgrid(torch.arange(0, H_img, device=device),
                           torch.arange(0, W_img, device=device), indexing='ij')
    ones = torch.ones_like(x)
    grid = torch.stack((x, y, ones), dim=0).float()  # (3, H_img, W_img)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)    # (B, 3, H_img, W_img)
    grid = grid.view(B, 3, -1)                         # (B, 3, H_img*W_img)
    
    # Apply sky mask to skip sky pixels
    sky_mask_flat = sky_mask.view(B, 1, -1)  # (B, 1, H_img*W_img)
    valid_mask = ~sky_mask_flat.bool()  # (B, 1, H_img*W_img)
    
    # Convert pixel coordinates to camera coordinates: P = depth * K^-1 * pixel
    K_inv = torch.inverse(K)  # (B, 3, 3)
    cam_coords = torch.bmm(K_inv, grid)  # (B, 3, H_img*W_img)
    cam_coords = cam_coords * depth_t.view(B, 1, -1)  # (B, 3, H_img*W_img)
    
    # Apply valid mask to skip sky pixels
    # cam_coords = cam_coords * valid_mask.float()  # (B, 3, H_img*W_img)


    # Convert to homogeneous coordinates (B, 4, H_img*W_img)
    ones_h = torch.ones(B, 1, cam_coords.shape[2], device=device)
    cam_coords_h = torch.cat([cam_coords, ones_h], dim=1)  # (B, 4, H_img*W_img)

    # Transform points from coordinate system at t to t'
    cam_coords_tprime_h = torch.bmm(T_t_to_tprime, cam_coords_h)  # (B, 4, H_img*W_img)
    cam_coords_tprime = cam_coords_tprime_h[:, :3, :]  # (B, 3, H_img*W_img)

    # Project the 3D points at t' to pixel coordinates using intrinsics K
    proj_coords = torch.bmm(K, cam_coords_tprime)  # (B, 3, H_img*W_img)
    eps = 1e-7
    proj_coords = proj_coords / (proj_coords[:, 2:3, :] + eps)  # (B, 3, H_img*W_img)

    # Reshape and normalize pixel coordinates to [-1, 1] for grid_sample
    x_proj = proj_coords[:, 0, :].view(B, H_img, W_img)
    y_proj = proj_coords[:, 1, :].view(B, H_img, W_img)
    x_norm = 2.0 * (x_proj / (W_img - 1)) - 1.0
    y_norm = 2.0 * (y_proj / (H_img - 1)) - 1.0
    warp_grid = torch.stack((x_norm, y_norm), dim=-1)  # (B, H_img, W_img, 2)

    # Generate a visible mask based on whether the projected coordinates lie within [-1, 1]
    visible_mask = ((x_norm > -1.0) & (x_norm < 1.0) & (y_norm > -1.0) & (y_norm < 1.0)).float().unsqueeze(1)  # (B, 1, H_img, W_img)
    visible_mask = visible_mask*valid_mask.view(B, 1, H_img, W_img)
    for b in range(visible_mask.shape[0]):  # Iterate over the batch
        mask_np = visible_mask[b, 0].cpu().numpy()  # Convert to NumPy for processing
        interpolated_mask = scipy.ndimage.gaussian_filter(mask_np, sigma=5)  # Apply Gaussian blur
        interpolated_mask = (interpolated_mask > 0.5).astype(float)  # Threshold to binary
        visible_mask[b, 0] = torch.tensor(interpolated_mask, device=visible_mask.device)

    # unvisible_mask = (1 - visible_mask)

    # Warp I_t to I_t' using grid_sample. Out-of-bound regions become zeros.
    I_tprime_warped = F.grid_sample(I_t, warp_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return I_tprime_warped.squeeze().cpu().numpy(), visible_mask.squeeze().cpu().numpy()

def compute_relative_pose(R1, T1, R2, T2):
    """
    Compute the relative pose (R, T) from camera t1 to camera t2.

    Args:
    - R1, T1: Rotation matrix (3x3) and translation vector (3,) for image t1.
    - R2, T2: Rotation matrix (3x3) and translation vector (3,) for image t2.

    Returns:
    - E_relative: Relative pose as a 4x4 matrix.
    """
    # Compute the inverse of E1
    R1_inv = R1.T  # Transpose of rotation matrix
    T1_inv = -np.dot(R1_inv, T1)

    # Compute relative rotation
    R_relative = np.dot(R2, R1_inv)

    # Compute relative translation
    # T_relative = T2 - np.dot(R_relative, T1)
    T_relative = T2 - T1

    # Construct the relative pose matrix
    E_relative = np.eye(4)
    E_relative[:3, :3] = R_relative
    E_relative[:3, 3] = T_relative

    return E_relative

def slerp(R1, R2, alpha):
    """
    Perform Spherical Linear Interpolation (SLERP) between two quaternions.
    
    Parameters:
        q1 (ndarray): Starting quaternion, shape (4,).
        q2 (ndarray): Ending quaternion, shape (4,).
        alpha (float): Interpolation factor in [0, 1].
        
    Returns:
        ndarray: Interpolated quaternion, shape (4,).
    """
    q1 = R.from_matrix(R1).as_quat()
    q2 = R.from_matrix(R2).as_quat()
    # Normalize quaternions (to avoid numerical issues)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute the dot product (cosine of the angle between quaternions)
    dot_product = np.dot(q1, q2)

    # Ensure the shortest path by flipping q2 if necessary
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product

    # Clamp dot product to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute the angle between quaternions
    theta = np.arccos(dot_product)

    # If the angle is small, use linear interpolation to avoid division by zero
    if theta < 1e-6:
        return alpha * q1 + (1-alpha) * q2

    # Perform SLERP
    sin_theta = np.sin(theta)
    q_interp = (np.sin(alpha * theta) / sin_theta) * q1 + (np.sin((1-alpha) * theta) / sin_theta) * q2
    rotation = R.from_quat(q_interp)

    # Convert to rotation matrix (3x3)
    R_matrix = rotation.as_matrix()

    return R_matrix

def interpolate_ego_pose(ego_pose_t1, ego_pose_t2, alpha=None):
    """
    Interpolate the ego_pose between two timestamps using random alpha between 0 and 1.
    
    ego_pose_t1: ego_pose at timestamp t (4x4 matrix)
    ego_pose_t2: ego_pose at timestamp t+1 (4x4 matrix)
    alpha: interpolation factor (random between 0 and 1 if None)
    """
    # Extract rotation matrix (top-left 3x3) and translation vector (top-right 3x1)
    R_t1 = ego_pose_t1[:3, :3]
    T_t1 = ego_pose_t1[:3, 3]
    
    R_t2 = ego_pose_t2[:3, :3]
    T_t2 = ego_pose_t2[:3, 3]

    # Interpolate the rotation quaternion using SLERP
    R_interpolated = slerp(R_t1, R_t2, alpha)

    # Interpolate the translation linearly
    T_interpolated = T_t1*alpha + (1-alpha) * T_t2

    # Reconstruct the interpolated ego_pose matrix
    ego_pose_interpolated = np.eye(4)
    ego_pose_interpolated[:3, :3] = R_interpolated
    ego_pose_interpolated[:3, 3] = T_interpolated

    return ego_pose_interpolated

def generate_pseudo_metadata(metadata_t1, metadata_t2, alpha): 
    pseudo_metadata = dict()
    pseudo_metadata['frame'] = metadata_t1['frame']*alpha + (alpha-1)*metadata_t2['frame']
    pseudo_metadata['cam'] = metadata_t1['cam']
    pseudo_metadata['frame_idx'] = metadata_t1['frame_idx']*alpha + (alpha-1)*metadata_t2['frame_idx']
    pseudo_metadata['ego_pose'] = interpolate_ego_pose(metadata_t1['ego_pose'], metadata_t2['ego_pose'], alpha)
    pseudo_metadata['extrinsic'] = metadata_t1['extrinsic']
    pseudo_metadata['timestamp'] = metadata_t1['timestamp']*alpha + (alpha-1)*metadata_t2['timestamp']
    pseudo_metadata['is_val'] = False
    return pseudo_metadata

def generate_pseudo_view(cam_info_t1, cam_info_t2): 
    alpha = np.random.rand() * (1-1e-10) + 1e-10
    # alpha = 0.5
    uid = cam_info_t1.uid*alpha + (1-alpha)*cam_info_t2.uid
    raw_time = cam_info_t1.raw_time*alpha + (1-alpha)*cam_info_t2.raw_time
    time = cam_info_t1.time*alpha + (1-alpha)*cam_info_t2.time
    FovX = cam_info_t1.FovX
    FovY = cam_info_t1.FovY
    K = cam_info_t1.K
    # width = cam_info_t1.width
    # height = cam_info_t1.height
    width = 1600
    height = 1066
    pseudo_metadata = generate_pseudo_metadata(cam_info_t1.metadata, cam_info_t2.metadata, alpha)
    pseudo_T = cam_info_t1.T*alpha + (1-alpha)*cam_info_t2.T
    pseudo_R = slerp(cam_info_t1.R, cam_info_t2.R, alpha=alpha)
    
    ego_R_t1 = cam_info_t1.metadata['ego_pose'][:3, :3]
    ego_T_t1 = cam_info_t1.metadata['ego_pose'][:3, 3]
    
    ego_R_t2 = cam_info_t2.metadata['ego_pose'][:3, :3]
    ego_T_t2 = cam_info_t2.metadata['ego_pose'][:3, 3]
    
    ego_R_pseudo = pseudo_metadata['ego_pose'][:3, :3]
    ego_T_pseudo = pseudo_metadata['ego_pose'][:3, 3]

    # pseudo_metadata["relative_pose_t1"] = compute_relative_pose(ego_R_t1, ego_T_t1, ego_R_pseudo, ego_T_pseudo)
    # pseudo_metadata["relative_pose_t2"] = compute_relative_pose(ego_R_t2, ego_T_t2, ego_R_pseudo, ego_T_pseudo)
    # pseudo_metadata["relative_pose_t1_t2"] = compute_relative_pose(ego_R_t1, ego_T_t1, ego_R_t2, ego_T_t2)

    
    pseudo_metadata["relative_pose_t1"] = compute_relative_pose(cam_info_t1.R, cam_info_t1.T, pseudo_R, pseudo_T)
    pseudo_metadata["relative_pose_t2"] = compute_relative_pose(cam_info_t2.R, cam_info_t2.T, pseudo_R, pseudo_T)
    # pseudo_metadata["relative_pose_t1_t2"] = compute_relative_pose(cam_info_t1.R, cam_info_t1.T, cam_info_t2.R, cam_info_t2.T)

    
    pseudo_metadata["image_t1"] = cam_info_t1.image
    pseudo_metadata["image_t2"] = cam_info_t2.image
    
    lidar_depth_t1 = cam_info_t1.metadata['lidar_depth']
    lidar_depth_t1 = lidar_depth_t1[lidar_depth_t1 > 0]
    # pseudo_metadata["depth_t1"] = cam_info_t1.metadata['pseudo_depth'] * (np.median(lidar_depth_t1) / np.median(cam_info_t1.metadata['pseudo_depth']))
    pseudo_metadata["depth_t1"] = cam_info_t1.metadata['pseudo_depth'] 
    pseudo_metadata["depth_t2"] = cam_info_t2.metadata['pseudo_depth'] 

    # lidar_depth_t2 = cam_info_t2.metadata['lidar_depth']
    # lidar_depth_t2 = lidar_depth_t2[lidar_depth_t2 > 0]
    # pseudo_metadata["depth_t2"] = cam_info_t2.metadata['pseudo_depth'] * (np.median(lidar_depth_t2) / np.median(cam_info_t2.metadata['pseudo_depth']))
    # pseudo_metadata["depth_t2"] = cam_info_t2.metadata['pseudo_depth'] 

    pseudo_metadata["K_t1"] = cam_info_t1.K.copy()
    # pseudo_metadata["K_t1"][0, :] *= (1600 / 1920) # width
    # pseudo_metadata["K_t1"][1, :] *= (1066 / 1280) # height
    pseudo_metadata["K_t1"][0, :] *= (width / 1920) # width
    pseudo_metadata["K_t1"][1, :] *= (height / 1280) # height

    # pseudo_metadata["K_t2"] = cam_info_t2.K.copy()
    # pseudo_metadata["K_t2"][0, :] *= (1600 / 1920)
    # pseudo_metadata["K_t2"][1, :] *= (1066 / 1280)
    
    pseudo_metadata["inv_K_t1"] = np.linalg.pinv(pseudo_metadata["K_t1"])
    # pseudo_metadata["inv_K_t2"] = np.linalg.pinv(pseudo_metadata["K_t2"])
    
    semantic_mask_t1 = torch.tensor(cam_info_t1.metadata['semantic']).unsqueeze(0).unsqueeze(0).float() # Convert (H, W) to (1, 1, H, W)
    semantic_mask_resized_t1 = F.interpolate(semantic_mask_t1, size=pseudo_metadata["depth_t1"].shape, mode='bilinear', align_corners=False)
    sky_mask_t1 = semantic_mask_resized_t1 == 10 # 10 is class_idx of sky class
    I_t1_warp, mask_t1 = warp_I_t_to_I_tprime_with_mask(cam_info_t1.image, pseudo_metadata["depth_t1"], pseudo_metadata["relative_pose_t1"], K, sky_mask_t1)
    
    semantic_mask_t2 = torch.tensor(cam_info_t1.metadata['semantic']).unsqueeze(0).unsqueeze(0).float() # Convert (H, W) to (1, 1, H, W)
    semantic_mask_resized_t2 = F.interpolate(semantic_mask_t2, size=pseudo_metadata["depth_t2"].shape, mode='bilinear', align_corners=False)
    sky_mask_t2 = semantic_mask_resized_t2 == 10 # 10 is class_idx of sky class
    I_t2_warp, mask_t2 = warp_I_t_to_I_tprime_with_mask(cam_info_t2.image, pseudo_metadata["depth_t2"], pseudo_metadata["relative_pose_t2"], K, sky_mask_t2)
    
    # Squeeze the masks to remove the singleton dimension
    visible_mask_t1 = np.squeeze(mask_t1.astype(bool))
    visible_mask_t2 = np.squeeze(mask_t2.astype(bool))

    # Transpose the images to match the mask's shape
    I_t1_warp = np.transpose(I_t1_warp, (1, 2, 0))  # (1280, 1920, 3)
    I_t2_warp = np.transpose(I_t2_warp, (1, 2, 0))  # (1280, 1920, 3)

    # Combine the images using the visible masks
    combined_image = np.where(visible_mask_t1[..., None], I_t1_warp, I_t2_warp)

    # Define a threshold for the pixel difference
    threshold = 0.05

    # Calculate the absolute difference between the two images
    diff = np.abs(I_t1_warp - I_t2_warp)

    # Create a mask for the differences larger than the threshold
    diff_mask = (diff > threshold).any(axis=2)

    # Combine the masks
    final_mask = visible_mask_t1 & visible_mask_t2 & ~diff_mask
    
    # Apply the final mask to the combined image
    # final_image = combined_image * final_mask[..., None]
    final_image = combined_image
    

    # pseudo_cam_info = PseudoCameraInfo(
    #         uid=uid, R=pseudo_R, T=pseudo_T, FovY=FovY, FovX=FovX, K=K,
    #         # image=cam_info_t1.image.resize((width, height)), 
    #         image_t1 = I_t1_warp,
    #         image_t2 = I_t2_warp,
    #         image_path=None, image_name=str(uid),
    #         width=width, height=height, 
    #         mask=None,
    #         metadata=pseudo_metadata,
    #         raw_time = raw_time, 
    #         time=time)

# Convert to PIL Image
    # pseudo_cam_info = CameraInfo(
    #         uid=uid, R=pseudo_R, T=pseudo_T, FovY=FovY, FovX=FovX, K=K,
    #         # image=cam_info_t1.image.resize((width, height)), 
    #         image = Image.fromarray(np.transpose((final_image * 255).astype(np.uint8), (1, 2, 0))),
    #         image_path=None, image_name=str(uid),
    #         width=width, height=height, 
    #         mask=Image.fromarray(~final_mask*255),
    #         metadata=pseudo_metadata,
    #         raw_time = raw_time, 
    #         time=time) 
    
    pseudo_cam_info = CameraInfo(
            uid=uid, R=pseudo_R, T=pseudo_T, FovY=FovY, FovX=FovX, K=K,
            # image=cam_info_t1.image.resize((width, height)), 
            image = Image.fromarray((final_image * 255).astype(np.uint8)),
            image_path=None, image_name=str(uid),
            width=width, height=height, 
            mask=Image.fromarray((final_mask).astype(np.uint8)*255),
            metadata=pseudo_metadata,
            raw_time = raw_time, 
            time=time) 
    
    
    return pseudo_cam_info

def my_generate_bird_eye_views(cam_infos, height_above_scene=10.0):
    bird_eye_cams = []

    for cam_info in cam_infos:
        existing_metadata = cam_info.metadata
        
        bird_eye_metadata = dict()
        bird_eye_metadata['frame'] = existing_metadata['frame']
        bird_eye_metadata['cam'] = existing_metadata['cam']
        bird_eye_metadata['frame_idx'] = existing_metadata['frame_idx']
        bird_eye_metadata['timestamp'] = existing_metadata['timestamp']
        bird_eye_metadata['is_val'] = existing_metadata['is_val']
        bird_eye_metadata['obj_bound'] = existing_metadata['obj_bound']

        translation = existing_metadata['ego_pose'][:3, 3] 

        bird_eye_position = np.array([translation[0], translation[1], height_above_scene])

        rotation_matrix = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, -1]])  

        bird_eye_extrinsic = np.eye(4) 
        bird_eye_extrinsic[:3, :3] = rotation_matrix  
        bird_eye_extrinsic[:3, 3] = bird_eye_position 

        cam_info.metadata['extrinsic'] = bird_eye_extrinsic
        cam_info.metadata['ego_pose'] = bird_eye_extrinsic
        # cam_info.metadata =bird_eye_metadata

        bird_eye_cams.append(cam_info)

    return bird_eye_cams

def readWaymoFullInfo(path, images='images', split_train=-1, split_test=-1, **kwargs):
    selected_frames = cfg.data.get('selected_frames', None)
    if cfg.debug:
        selected_frames = [0, 0]
    if cfg.data.get('load_pcd_from', False) and (cfg.mode == 'train'):
        load_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'input_ply')
        save_dir = os.path.join(cfg.model_path, 'input_ply')
        os.system(f'rm -rf {save_dir}')
        shutil.copytree(load_dir, save_dir)
        
        colmap_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'colmap')
        save_dir = os.path.join(cfg.model_path, 'colmap')
        os.system(f'rm -rf {save_dir}')
        shutil.copytree(colmap_dir, save_dir)
        
    bkgd_ply_path = os.path.join(cfg.model_path, 'input_ply/points3D_bkgd.ply')
    pretrain_ply_path = os.path.join(os.path.join(cfg.pretrain_model_path, cfg.model_path.split('/')[-1]), 'input_ply/points3D_bkgd.ply')
    # build_pointcloud = (cfg.mode == 'train') and (not os.path.exists(pretrain_ply_path) or cfg.data.get('regenerate_pcd', False))
    build_pointcloud = (cfg.mode == 'train') and (not os.path.exists(bkgd_ply_path) or cfg.data.get('regenerate_pcd', False))
    # dynamic mask
    dynamic_mask_dir = os.path.join(path, 'dynamic_mask')
    load_dynamic_mask = True

    # sky mask
    sky_mask_dir = os.path.join(path, 'sky_mask')
    if not os.path.exists(sky_mask_dir):
        cmd = f'python script/waymo/generate_sky_mask.py --datadir {path}'
        print('Generating sky mask')
        os.system(cmd)
    load_sky_mask = (cfg.mode == 'train')
    
    # lidar depth
    lidar_depth_dir = os.path.join(path, 'lidar_depth')
    if not os.path.exists(lidar_depth_dir):
        cmd = f'python script/waymo/generate_lidar_depth.py --datadir {path}'
        print('Generating lidar depth')
        os.system(cmd)
    # load_lidar_depth = (cfg.mode == 'train')
    load_lidar_depth = True
    load_pseudo_seg = cfg.data.use_semantic
    load_pseudo_depth = cfg.data.use_pseudo_depth
    
    # Optional: monocular normal cue
    mono_normal_dir = os.path.join(path, 'mono_normal')
    load_mono_normal = cfg.data.use_mono_normal and (cfg.mode == 'train') and os.path.exists(mono_normal_dir)
        
    # Optional: monocular depth cue
    mono_depth_dir = os.path.join(path, 'mono_depth')
    load_mono_depth = cfg.data.use_mono_depth and (cfg.mode == 'train') and os.path.exists(mono_depth_dir)
    output = generate_dataparser_outputs(
        datadir=path, 
        selected_frames=selected_frames,
        build_pointcloud=build_pointcloud,
        cameras=cfg.data.get('cameras', [0, 1, 2]),
    )

    exts = output['exts']
    ixts = output['ixts']
    poses = output['poses']
    c2ws = output['c2ws']
    image_filenames = output['image_filenames']
    obj_tracklets = output['obj_tracklets']
    obj_info = output['obj_info']
    frames, cams = output['frames'], output['cams']
    frames_idx = output['frames_idx']
    num_frames = output['num_frames']
    cams_timestamps = output['cams_timestamps']
    tracklet_timestamps = output['tracklet_timestamps']
    obj_bounds = output['obj_bounds']
    train_frames, test_frames = get_val_frames(
        num_frames, 
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    scene_metadata = dict()
    scene_metadata['obj_tracklets'] = obj_tracklets
    scene_metadata['tracklet_timestamps'] = tracklet_timestamps
    scene_metadata['obj_meta'] = obj_info
    scene_metadata['num_images'] = len(exts)
    scene_metadata['num_cams'] = len(cfg.data.cameras)
    scene_metadata['num_frames'] = num_frames
    
    camera_timestamps = dict()
    for cam in cfg.data.get('cameras', [0, 1, 2]):
        camera_timestamps[cam] = dict()
        camera_timestamps[cam]['train_timestamps'] = []
        camera_timestamps[cam]['test_timestamps'] = []      

    ########################################################################################################################
    timestamp_mapper = {}
    time_line = [i for i in range(selected_frames[0], selected_frames[-1] + 1)]
    time_length = selected_frames[-1] - selected_frames[0]
    for index, time in enumerate(time_line):
        # timestamp_mapper[time] = (time-selected_frames[0])/time_length
        timestamp_mapper[time] = -1 + 2*((time-selected_frames[0])/time_length) # normalize to [-1, 1] for appropriate with pvg


    cam_infos = []
    pseudo_cam_infos = []
    for i in tqdm(range(len(exts))):
        # generate pose and image
        ext = exts[i]
        ixt = ixts[i]
        c2w = c2ws[i]
        pose = poses[i]
        image_path = image_filenames[i]
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path)

        width, height = image.size
        fx, fy = ixt[0, 0], ixt[1, 1]
        FovY = focal2fov(fx, height)
        FovX = focal2fov(fy, width)    
        
        # if cfg.render.coord == 'world':
        #     RT = np.linalg.inv(c2w)        # render in world space
        # else:
        #     RT = np.linalg.inv(ext)        # render in vehicle space
        RT = np.linalg.inv(c2w)
        R = RT[:3, :3].T
        T = RT[:3, 3]
        K = ixt.copy()
        
        metadata = dict()
        metadata['frame'] = frames[i]
        metadata['cam'] = cams[i]
        metadata['frame_idx'] = frames_idx[i]
        metadata['ego_pose'] = pose
        metadata['extrinsic'] = ext
        metadata['timestamp'] = cams_timestamps[i]
        time = timestamp_mapper[frames[i]]

        if frames_idx[i] in train_frames:
            metadata['is_val'] = False
            camera_timestamps[cams[i]]['train_timestamps'].append(cams_timestamps[i])
        else:
            metadata['is_val'] = True
            camera_timestamps[cams[i]]['test_timestamps'].append(cams_timestamps[i])
        
        # load dynamic mask
        if load_dynamic_mask:
            # dynamic_mask_path = os.path.join(dynamic_mask_dir, f'{image_name}.png')
            # obj_bound = (cv2.imread(dynamic_mask_path)[..., 0]) > 0.
            # obj_bound = Image.fromarray(obj_bound)
            metadata['obj_bound'] = Image.fromarray(obj_bounds[i])
                    
        # load lidar depth
        if load_lidar_depth:
            depth_path = os.path.join(path, 'lidar_depth', f'{image_name}.npy')
            
            depth = np.load(depth_path, allow_pickle=True)
            if isinstance(depth, np.ndarray):
                depth = dict(depth.item())
                mask = depth['mask']
                value = depth['value']
                depth = np.zeros_like(mask).astype(np.float32)
                depth[mask] = value
            metadata['lidar_depth'] = depth
            
        # load sky mask
        if load_sky_mask:
            sky_mask_path = os.path.join(sky_mask_dir, f'{image_name}.png')
            sky_mask = (cv2.imread(sky_mask_path)[..., 0]) > 0.
            sky_mask = Image.fromarray(sky_mask)
            metadata['sky_mask'] = sky_mask
        
        # Optional: load monocular normal
        if load_mono_normal:
            mono_normal_path = os.path.join(mono_normal_dir, f'{image_name}.npy')
            mono_normal = np.load(mono_normal_path)
            metadata['mono_normal'] = mono_normal

        # Optional load midas depth
        if load_mono_depth:
            mono_depth_path = os.path.join(mono_depth_dir, f'{image_name}.npy')
            mono_depth = np.load(mono_depth_path)
            metadata['mono_depth'] = mono_depth
        
        # load pseud segmentation
        if load_pseudo_seg: 
            seg_path = os.path.join(path, 'pseudo_segs', f'{image_name}.png')
            seg_map = np.array(Image.open(seg_path))
            metadata['semantic'] = seg_map
        
        if load_pseudo_depth: 
            pseudo_depth_path = os.path.join(path, 'pseudo_depth', f'{image_name}.npy')
            pseudo_depth = np.load(pseudo_depth_path)
            # pseudo_depth_min = pseudo_depth.min()
            # pseudo_depth_max = pseudo_depth.max()
            # pseudo_depth_normalized = (pseudo_depth - pseudo_depth_min) / (pseudo_depth_max - pseudo_depth_min)
            # min_depth = 0.001
            # max_depth = 1000.0
            # A = (1 / min_depth) - (1/ max_depth)
            # B = 1 / max_depth
            # midas_depth_aligned = 1 / (A * pseudo_depth_normalized + B)
            metadata['pseudo_depth'] = pseudo_depth.squeeze()
        
        # Load pseudo optical flow
        # if True: 
        #     pseudo_optical_flow_path = os.path.join(path, 'pseudo_optical_flow', f'{image_name}.npy')
        #     if os.path.exists(pseudo_optical_flow_path):
        #         pseudo_optical_flow = np.load(pseudo_optical_flow_path)
        #         metadata['pseudo_optical_flow'] = pseudo_optical_flow.squeeze()

        mask = None
        # change to BEV
        # R = np.array([[1, 0, 0],[0, 1, 0],[0, 0, -1]])  
        # T[-1] = 20    
        cam_info = CameraInfo(
            uid=i, R=R, T=T, FovY=FovY, FovX=FovX, K=K,
            image=image, image_path=image_path, image_name=image_name,
            width=width, height=height, 
            mask=mask,
            metadata=metadata,
            raw_time = frames[i], 
            time=time)
        cam_infos.append(cam_info)
        
    # if cfg.data.gen_pseudo_view: 
        # sys.stdout.write('\n')
    # cam_infos = my_generate_bird_eye_views(cam_infos)
    train_cam_infos = [cam_info for cam_info in cam_infos if not cam_info.metadata['is_val']]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata['is_val']]
    if cfg.data.generate_pseudo_view:
        for idx in range(len(train_cam_infos)-1): 
            for j in range(3):
                pseudo_cam_info = generate_pseudo_view(train_cam_infos[idx], train_cam_infos[idx+1])
                pseudo_cam_infos.append(pseudo_cam_info)
        pseudo_cam_infos = [cam_info for cam_info in pseudo_cam_infos]
    
    for cam in cfg.data.get('cameras', [0, 1, 2]):
        camera_timestamps[cam]['train_timestamps'] = sorted(camera_timestamps[cam]['train_timestamps'])
        camera_timestamps[cam]['test_timestamps'] = sorted(camera_timestamps[cam]['test_timestamps'])
    scene_metadata['camera_timestamps'] = camera_timestamps
        
    novel_view_cam_infos = []
    
    #######################################################################################################################3
    # Get scene extent
    # 1. Default nerf++ setting
    if cfg.mode == 'novel_view':
        nerf_normalization = getNerfppNorm(novel_view_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)

    # 2. The radius we obtain should not be too small (larger than 10 here)
    nerf_normalization['radius'] = max(nerf_normalization['radius'], 10)
    
    # 3. If we have extent set in config, we ignore previous setting
    if cfg.data.get('extent', False):
        nerf_normalization['radius'] = cfg.data.extent
    
    # 4. We write scene radius back to config
    cfg.data.extent = float(nerf_normalization['radius'])

    # 5. We write scene center and radius to scene metadata    
    scene_metadata['scene_center'] = nerf_normalization['center']
    scene_metadata['scene_radius'] = nerf_normalization['radius']
    print(f'Scene extent: {nerf_normalization["radius"]}')

    # Get sphere center
    lidar_ply_path = os.path.join(cfg.model_path, 'input_ply/points3D_lidar.ply')
    if os.path.exists(lidar_ply_path):
        sphere_pcd: BasicPointCloud = fetchPly(lidar_ply_path)
    else:
        sphere_pcd: BasicPointCloud = fetchPly(bkgd_ply_path)
    
    sphere_normalization = get_Sphere_Norm(sphere_pcd.points)
    scene_metadata['sphere_center'] = sphere_normalization['center']
    scene_metadata['sphere_radius'] = sphere_normalization['radius']
    print(f'Sphere extent: {sphere_normalization["radius"]}')
    pcd: BasicPointCloud = fetchPly(bkgd_ply_path)
    if cfg.mode == 'train':
        point_cloud = pcd
    else:
        point_cloud = None
        bkgd_ply_path = None

    scene_info = SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        pseudo_train_cameras=pseudo_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=bkgd_ply_path,
        metadata=scene_metadata,
        novel_view_cameras=novel_view_cam_infos,
    )
    
    return scene_info
    
    
    
