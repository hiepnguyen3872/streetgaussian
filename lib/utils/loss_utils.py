#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from math import exp
from lib.utils.img_utils import save_img_torch
# from lib.config import cfg
from lib.datasets.project_depth import *

from PIL import Image
from lib.utils.general_utils import NumpytoTorch
from lib.utils.general_utils import PILtoTorch


def l1_loss(network_output, gt, mask=None):
    '''
    network_output, gt: (C, H, W)
    mask: (1, H, W) 
    '''

    network_output = network_output.permute(1, 2, 0) # [H, W, C]
    gt = gt.permute(1, 2, 0) # [H, W, C]

    if mask is not None:
        mask = mask.squeeze(0) # [H, W]
        network_output = network_output[mask]
        gt = gt[mask]
    
    loss = ((torch.abs(network_output - gt))).mean()

    return loss

def l2_loss(network_output, gt, mask=None):
    '''
    network_output, gt: (C, H, W)
    mask: (1, H, W) 
    '''
    
    network_output = network_output.permute(1, 2, 0) # [H, W, C]
    gt = gt.permute(1, 2, 0) # [H, W, C]    
    
    if mask is not None:
        mask = mask.squeeze(0) # [H, W]
        network_output = network_output[mask]
        gt = gt[mask]

    loss =  (((network_output - gt) ** 2)).mean()

    return loss

    
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask=None):
    '''
    img1, img2: (C, H, W)
    mask: (1, H, W)
    '''    
    
    img1 = img1.permute(1, 2, 0)
    img2 = img2.permute(1, 2, 0)
    
    if mask is not None:
        mask = mask.squeeze(0)
        img1 = img1[mask]
        img2 = img2[mask]
    
    # mse = ((img1 - img2) ** 2).view(-1, img1.shape[-1]).mean(dim=0, keepdim=True)    
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr
    
    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    
    if mask is not None:
        img1 = torch.where(mask, img1, torch.zeros_like(img1))
        img2 = torch.where(mask, img2, torch.zeros_like(img2))
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())

    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def silog_loss(prediction, target, variance_focus: float = 0.85) -> float:
    """
    Compute SILog loss. See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for
    more information about scale-invariant loss.

    Args:
        prediction (Tensor): Prediction.
        target (Tensor): Target.
        variance_focus (float): Variance focus for the SILog computation.

    Returns:
        float: SILog loss.
    """

    # let's only compute the loss on non-null pixels from the ground-truth depth-map
    non_zero_mask = (target > 0) & (prediction > 0)

    # SILog
    d = torch.log(prediction[non_zero_mask]) - torch.log(target[non_zero_mask])
    return torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0

def smooth_loss(motion_mask, pseudo_depth):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_motion_mask_x = torch.abs(motion_mask[:, :, :-1] - motion_mask[:, :, 1:])
    grad_motion_mask_y = torch.abs(motion_mask[:, :-1, :] - motion_mask[:, 1:, :])

    grad_pseudo_depth_x = torch.mean(torch.abs(pseudo_depth[:, :, :-1] - pseudo_depth[:, :, 1:]), 1, keepdim=True)
    grad_pseudo_depth_y = torch.mean(torch.abs(pseudo_depth[:, :-1, :] - pseudo_depth[:, 1:, :]), 1, keepdim=True)

    grad_motion_mask_x *= torch.exp(-grad_pseudo_depth_x)
    grad_motion_mask_y *= torch.exp(-grad_pseudo_depth_y)

    return grad_motion_mask_x.mean() + grad_motion_mask_y.mean()

def get_dynamic_mask(seg_map): 
    class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                       'traffic light', 'traffic sign', 'vegetation', 'terrain', 
                       'sky', 'person', 'rider', 'car', 'truck', 'bus', 
                       'train', 'motorcycle', 'bicycle']
        
    dynamic_classes = {'person', 'rider', 'car', 'truck', 'bus', 
                           'train', 'motorcycle', 'bicycle'}
                
    dynamic_class_indices = [class_names.index(cls) for cls in dynamic_classes]
        
    # Create dynamic mask
    dynamic_mask = torch.isin(seg_map, torch.tensor(dynamic_class_indices, device=seg_map.device))
    
    return dynamic_mask

def warp_reconstruction_loss(pseudo_view, render, lambda_l1, lambda_dssim): 
    crop_resolution_height = 533
    crop_resolution_width = 800
    resolution = (pseudo_view.image_width, pseudo_view.image_height)
    target_image = PILtoTorch(pseudo_view.meta['image_t1'], resolution, resize_mode=Image.BILINEAR)[:3, ...]
    target_image = target_image.to(render["rgb"].device)
    
    
    depth_t1 = NumpytoTorch(pseudo_view.meta['depth_t1'], resolution, resize_mode=Image.NEAREST).to(render["rgb"].device)
    depth_t1 = depth_t1.cpu().numpy()
    cam_points = back_project_depth(height = pseudo_view.image_height, width=pseudo_view.image_width, depth=depth_t1, inv_K=pseudo_view.meta['inv_K_t1'])

    
    pix_coords = project_3D(height=pseudo_view.image_height, width=pseudo_view.image_width, points=cam_points, K=pseudo_view.meta['K_t1'], T=pseudo_view.meta['relative_pose_t1'])
    pix_coords = pix_coords.to(render["rgb"].device)
    
    reconstruct_img_t1 = F.grid_sample(render["rgb"].unsqueeze(0),pix_coords.unsqueeze(0),padding_mode="border").squeeze()
    
    center_height, center_width = target_image.shape[1] // 2, target_image.shape[2] // 2
    mask = torch.zeros_like(target_image[0], dtype=torch.bool)  # Initialize mask with False
    mask[center_height-crop_resolution_height//2:crop_resolution_height//2 + center_height, 
         center_width-crop_resolution_width//2:crop_resolution_width//2 + center_width] = True  # Set center region to True
    mask = mask.to(render["rgb"].device)
    Ll1 = l1_loss(reconstruct_img_t1, target_image, mask = mask).item()
    loss = (1.0 - lambda_dssim) * lambda_l1 * Ll1 + lambda_dssim * (1.0 - ssim(reconstruct_img_t1, target_image, mask=mask))
    # reconstruct_img_t1 = render["rgb"]
    # Ll1 = l1_loss(reconstruct_img_t1, target_image).item()
    # loss = (1.0 - lambda_dssim) * lambda_l1 * Ll1 + lambda_dssim * (1.0 - ssim(reconstruct_img_t1, target_image))
    # return loss, render["rgb"][:,mask].reshape(3, 256, 512), target_image[:,mask].reshape(3, 256, 512), reconstruct_img_t1[:,mask].reshape(3, 256, 512), depth_t1
    return loss, draw_line_box(render["rgb"].clone()), draw_line_box(target_image.clone()), draw_line_box(reconstruct_img_t1.clone()), depth_t1

def draw_line_box(tensor_img): 
    box_color = torch.tensor([1.0, 0.0, 0.0])  # Red color in RGB
    box_height, box_width = 533, 800
    center_height, center_width = tensor_img.shape[1] // 2, tensor_img.shape[2] // 2

    # # Draw the box on the reconstructed image
    # tensor_img[:, center_height - box_height // 2:center_height + box_height // 2, 
    #            center_width - box_width // 2:center_width + box_width // 2] = box_color.view(3, 1, 1)
    # # Draw the outline of the box on the reconstructed image
    # tensor_img[:, center_height - box_height // 2:center_height + box_height // 2, 
    #                    center_width - box_width // 2:center_width + box_width // 2] = box_color.view(3, 1, 1)  # Fill the box area

    # Draw the edges of the box
    tensor_img[:, center_height - box_height // 2, center_width - box_width // 2:center_width + box_width // 2] = box_color.view(3, 1)  # Top edge
    tensor_img[:, center_height + box_height // 2 - 1, center_width - box_width // 2:center_width + box_width // 2] = box_color.view(3, 1)  # Bottom edge
    tensor_img[:, center_height - box_height // 2:center_height + box_height // 2, center_width - box_width // 2] = box_color.view(3, 1)  # Left edge
    tensor_img[:, center_height - box_height // 2:center_height + box_height // 2, center_width + box_width // 2 - 1] = box_color.view(3, 1)  # Right edge

    return tensor_img

    
def save_img_torch(tensor, filename):
    """
    Save a PyTorch tensor as an image file.

    Args:
        tensor (Tensor): The input tensor to save. Should be in the format (C, H, W).
        filename (str): The filename to save the image as.
    """
    # Ensure the tensor is in the right format (C, H, W)
    if tensor.dim() == 3:
        # Convert the tensor to a PIL image
        # Denormalize if necessary (assuming values are in [0, 1])
        tensor = tensor.clamp(0, 1)  # Clamp values to [0, 1]
        tensor = tensor.permute(1, 2, 0)  # Change to (H, W, C)
        image = Image.fromarray((tensor.numpy() * 255).astype('uint8'))  # Convert to uint8
        image.save(filename)
    else:
        raise ValueError("Input tensor must be 3-dimensional (C, H, W).")

def visualize_depth(depth_array, filename):
    """
    Visualize and save the depth array as an image.

    Args:
        depth_array (ndarray): The depth array to visualize. Should be in the format (H, W).
        filename (str): The filename to save the depth image as.
    """
    # Normalize the depth array to the range [0, 1]
    depth_min = depth_array.min()
    depth_max = depth_array.max()
    depth_normalized = (depth_array - depth_min) / (depth_max - depth_min)  # Normalize to [0, 1]
    # Convert to a PIL image and save
    depth_image = Image.fromarray((np.squeeze(depth_normalized) * 255).astype('uint8'))  # Convert to uint8
    depth_image.save(filename)


def temporal_smoothness_knn(positions, mu_t, sigma_t, k=10):
    """
    Regularize mu_t and sigma_t for spatially close Gaussians.
    Args:
        positions: [N, 3] tensor of 3D Gaussian centers
        mu_t: [N, 1] temporal center (learned parameter)
        sigma_t: [N, 1] temporal spread (learned parameter)
        k: number of nearest neighbors
    Returns:
        temporal smoothness loss (scalar)
    """
    with torch.no_grad():
        dist = torch.cdist(positions, positions)  # [N, N]
        knn_indices = dist.topk(k+1, largest=False).indices[:, 1:] 

    mu_t_neighbors = mu_t[knn_indices]         
    sigma_t_neighbors = sigma_t[knn_indices]  

    mu_t_i = mu_t.unsqueeze(1)
    sigma_t_i = sigma_t.unsqueeze(1)

    loss_mu = ((mu_t_i - mu_t_neighbors)**2).mean()
    loss_sigma = ((sigma_t_i - sigma_t_neighbors)**2).mean()

    return loss_mu, loss_sigma
