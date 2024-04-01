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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def prepare_single_gs(pc, time, stage='fine', xyz_offset=None, scales_preset=None, pre_scale=True):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception as e:
        # print(e)
        pass
    

    # means3D = pc.get_xyz
    means3D = pc.get_xyz
    if pre_scale:
        means3D = means3D * scales_preset
        means3D = xyz_offset(means3D)
    # add deformation to each points
    # deformation = pc.get_deformation
    try:
        assert time.item() >= 0 and time.item() <= 1
        time = time.to(means3D.device).repeat(means3D.shape[0],1)
    except:
        assert time >= 0 and time <= 1
        time = torch.tensor([time]).to(means3D.device).repeat(means3D.shape[0],1)
    # time = time / 16 # in range of [0, 1]

    means2D = screenspace_points
    opacity = pc._opacity
    color=pc._features_dc
    color=color[:,0,:]
    
    
    
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    # cov3D_precomp = None

    dx = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    # scales = pc.get_scaling
    scales = pc._scaling
    if scales.shape[-1] == 1:
        scales = scales.repeat(1, 3)
    #scales = torch.ones_like(scales ) * 0.03
    # rotations = pc.get_rotation
    rotations = pc._rotation
    deformation_point = pc._deformation_table
    # print('color render:',color.shape)   #[40000, 1, 3]->[40000, 3]
    # print('rotations render:',rotations.shape)  #[40000, 4]
    
    if stage == "static": # or time.sum() == 0:
    # if stage == "static" or time.sum() == 0:
        means3D_deform, scales_deform, rotations_deform, opacity_deform,color_deform = means3D, scales, rotations, opacity,color
    else:
        means3D_deform, scales_deform, rotations_deform, opacity_deform,color_deform = pc._deformation(means3D[deformation_point].detach(), scales[deformation_point].detach(), rotations[deformation_point].detach(), opacity[deformation_point].detach(),color[deformation_point].detach(), time[deformation_point].detach())
        # dx = (means3D_deform - means3D[deformation_point].detach())
        # ds = (scales_deform - scales[deformation_point].detach())
        # dr = (rotations_deform - rotations[deformation_point].detach())
        # do = (opacity_deform - opacity[deformation_point].detach())
        # #dc=0
        # dc = (color_deform - color[deformation_point].detach())

        # dx = dx * (time ** 0.35)
        # # dx = dx * time
        # means3D_deform = dx + means3D[deformation_point].detach()

    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    color_final= torch.zeros_like(color)
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform

    # print('color_final shape before',color_final.shape)

    # print('color_final shape',color_final.shape)
    # print('color_deform shape',color_deform.shape)
    # print('deformation_point shape',deformation_point.shape)
    color_final[deformation_point] = color_deform

    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]
    color_final[~deformation_point] = color[~deformation_point]
    color_final=torch.unsqueeze(color_final, 1)  #[40000,  3]->[40000, 1, 3]
    
    scales_final = pc.scaling_activation(scales_final)
    #scales_final = torch.ones_like(scales_final ) * 0.01
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity)
    #color without activation
    
    if not pre_scale:
        means3D_final = means3D_final * scales_preset
        means3D_final = xyz_offset(means3D_final)
    dx = (means3D_final - means3D.detach())
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    #print('color_final devide',dc.device)
    rest = pc.get_features_rest
    shs = torch.cat((color_final, rest), dim=1)
    return means3D_final, means2D, shs, opacity, scales_final, rotations_final, screenspace_points, dx

def move(x, axis, time):
    x[axis:axis+1] = 0.5 + 0.2 * time.to(x.device)
    return x

def placeholder(idx, time):
    # if idx == 0:
    return torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda", requires_grad=False)
def placeholder2(idx, time):
    # if idx == 0:
    return torch.eye(4, dtype=torch.float32, device="cuda", requires_grad=False)
    # return 

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, time=torch.tensor([[0]]), scaling_modifier = 1.0, override_color = None, stage=None, render_flow=False, return_pts=False, offset=[], scales_list=[], pre_scale=False):
    # print(scaling_modifier)
    assert scaling_modifier == 1
    if stage is None:
        raise NotImplementedError
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    

    # Set up rasterization configuration
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=0,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D_final, means2D, shs, opacity, scales_final, rotations_final, screenspace_points, dx = [], [], [], [], [], [], [], []
    
    # zero_ts = (bg_color * 0).detach()
    # offset = [zero_ts, move(zero_ts, 0, time)]
    # 0 is y
    # print(scales_list)
    for i, _ in enumerate(pc):
        means3D_final_, means2D_, shs_, opacity_, scales_final_, rotations_final_, screenspace_points_, dx_ = prepare_single_gs(_, time, xyz_offset=offset[i], scales_preset=scales_list[i], pre_scale=pre_scale)
        means3D_final.append(means3D_final_)
        means2D.append(means2D_)
        shs.append(shs_)
        opacity.append(opacity_)
        scales_final.append(scales_final_)
        rotations_final.append(rotations_final_)
        screenspace_points.append(screenspace_points_)
        dx.append(dx_)

    means3D_final = torch.cat(means3D_final, dim=0)
    means2D = torch.cat(means2D, dim=0)
    shs = torch.cat(shs, dim=0)
    opacity = torch.cat(opacity, dim=0)
    scales_final = torch.cat(scales_final, dim=0)
    rotations_final = torch.cat(rotations_final, dim=0)
    screenspace_points = torch.cat(screenspace_points, dim=0)
    dx = torch.cat(dx, dim=0)
    # print('means3D_final', means3D_final.shape)
    # print('means2D', means2D.shape)
    # print('shs', shs.shape)
    # print('opacity', opacity.shape)
    # print('scales_final', scales_final.shape)
    # print('rotations_final', rotations_final.shape)
    # print('screenspace_points', screenspace_points.shape)
    # print('dx', dx.shape)

    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None
    )


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    res = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "alpha": alpha,
        "depth":depth,
    }
    # print(dx, time.sum(), stage)
    if dx is not None:
        res['dx'] = dx #.mean()
        # res['ds'] = ds #.mean()
        # res['dr'] = dr #.mean()
        # res['do'] = do #.mean()
        # res['dc'] = dc

    if return_pts:
        res['means3D'] = means3D_final
        res['means2D'] = means2D
        res['opacity_final'] = opacity_final
    return res