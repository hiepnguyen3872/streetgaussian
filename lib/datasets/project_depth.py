from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def back_project_depth(height, width, depth, inv_K):
    height = height
    width = width
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = nn.Parameter(torch.from_numpy(id_coords),requires_grad=False)
    ones = nn.Parameter(torch.ones(1, height * width),requires_grad=False)
    pix_coords = torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0)
    pix_coords = pix_coords.repeat(1, 1)
    pix_coords = nn.Parameter(torch.cat([pix_coords, ones], 0),
                                   requires_grad=False)
    
    inv_K_tensor = torch.from_numpy(inv_K.astype(np.float32)).to(pix_coords.device)  # Convert inv_K to a PyTorch tensor with float32
    cam_points = torch.matmul(inv_K_tensor[:3, :3], pix_coords)
    depth_tensor = torch.from_numpy(depth)
    cam_points = depth_tensor.view(1, -1) * cam_points
    cam_points = torch.cat([cam_points, ones], 0)
    return cam_points


def project_3D(height, width, points, K, T, eps=1e-7):

    height = height
    width = width
    K_tensor = torch.zeros((4, 4), device=points.device)  # Create a (4, 4) tensor filled with zeros
    K_tensor[:3, :3] = torch.from_numpy(K.astype(np.float32)).to(points.device)  # Set the top-left 3x3 to K
    T_tensor = torch.from_numpy(T.astype(np.float32)).to(points.device)  # Convert T to a PyTorch tensor
    P = torch.matmul(K_tensor, T_tensor)[:3, :]
    cam_points = torch.matmul(P, points)
    pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + eps)
    pix_coords = pix_coords.view(2, height, width)
    pix_coords = pix_coords.permute(1, 2, 0)
    pix_coords[..., 0] /= width - 1
    pix_coords[..., 1] /= height - 1
    pix_coords = (pix_coords - 0.5) * 2
    return pix_coords