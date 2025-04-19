import torch
import numpy as np
def calculate_gs_optical_flow(render_t_1, render_t_2): 
    # Gaussian parameters at t_1
    proj_2D_t_1 = render_t_1["proj_means_2D"]
    gs_per_pixel = render_t_1["gs_per_pixel"].long() 
    weight_per_gs_pixel = render_t_1["weight_per_gs_pixel"]
    x_mu = render_t_1["x_mu"]
    cov2D_inv_t_1 = render_t_1["conic_2D"].detach()

    # Gaussian parameters at t_2
    proj_2D_t_2 = render_t_2["proj_means_2D"]
    cov2D_inv_t_2 = render_t_2["conic_2D"]
    cov2D_t_2 = render_t_2["conic_2D_inv"]


    cov2D_t_2_mtx = torch.zeros([cov2D_t_2.shape[0], 2, 2]).cuda()
    cov2D_t_2_mtx[:, 0, 0] = cov2D_t_2[:, 0]
    cov2D_t_2_mtx[:, 0, 1] = cov2D_t_2[:, 1]
    cov2D_t_2_mtx[:, 1, 0] = cov2D_t_2[:, 1]
    cov2D_t_2_mtx[:, 1, 1] = cov2D_t_2[:, 2]

    cov2D_inv_t_1_mtx = torch.zeros([cov2D_inv_t_1.shape[0], 2, 2]).cuda()
    cov2D_inv_t_1_mtx[:, 0, 0] = cov2D_inv_t_1[:, 0]
    cov2D_inv_t_1_mtx[:, 0, 1] = cov2D_inv_t_1[:, 1]
    cov2D_inv_t_1_mtx[:, 1, 0] = cov2D_inv_t_1[:, 1]
    cov2D_inv_t_1_mtx[:, 1, 1] = cov2D_inv_t_1[:, 2]

    # B_t_2
    U_t_2 = torch.svd(cov2D_t_2_mtx)[0]
    S_t_2 = torch.svd(cov2D_t_2_mtx)[1]
    V_t_2 = torch.svd(cov2D_t_2_mtx)[2]
    B_t_2 = torch.bmm(torch.bmm(U_t_2, torch.diag_embed(S_t_2)**(1/2)), V_t_2.transpose(1,2))
    # B_t_1 ^(-1)
    U_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[0]
    S_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[1]
    V_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[2]
    B_inv_t_1 = torch.bmm(torch.bmm(U_inv_t_1, torch.diag_embed(S_inv_t_1)**(1/2)), V_inv_t_1.transpose(1,2))

    # calculate B_t_2*B_inv_t_1
    B_t_2_B_inv_t_1 = torch.bmm(B_t_2, B_inv_t_1)

    # calculate cov2D_t_2*cov2D_inv_t_1
    # cov2D_t_2cov2D_inv_t_1 = torch.zeros([cov2D_inv_t_2.shape[0],2,2]).cuda()
    # cov2D_t_2cov2D_inv_t_1[:, 0, 0] = cov2D_t_2[:, 0] * cov2D_inv_t_1[:, 0] + cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 1]
    # cov2D_t_2cov2D_inv_t_1[:, 0, 1] = cov2D_t_2[:, 0] * cov2D_inv_t_1[:, 1] + cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 2]
    # cov2D_t_2cov2D_inv_t_1[:, 1, 0] = cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 0] + cov2D_t_2[:, 2] * cov2D_inv_t_1[:, 1]
    # cov2D_t_2cov2D_inv_t_1[:, 1, 1] = cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 1] + cov2D_t_2[:, 2] * cov2D_inv_t_1[:, 2]

    # isotropic version of GaussianFlow
    #predicted_flow_by_gs = (proj_2D_next[gs_per_pixel] - proj_2D[gs_per_pixel].detach()) * weights.detach()

    # full formulation of GaussianFlow
    cov_multi = (B_t_2_B_inv_t_1[gs_per_pixel] @ x_mu.permute(0,2,3,1).unsqueeze(-1).detach()).squeeze()
    predicted_flow_by_gs = (cov_multi + proj_2D_t_2[gs_per_pixel] - proj_2D_t_1[gs_per_pixel].detach() - x_mu.permute(0,2,3,1).detach()) *  weight_per_gs_pixel.unsqueeze(-1).detach()
    return predicted_flow_by_gs.sum(0)





def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel




def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_uv = flow_uv.cpu().numpy()
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)