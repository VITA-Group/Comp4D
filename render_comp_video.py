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
import imageio
import numpy as np
import torch
from scene.comp_scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.comp_renderer import render
# import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams,OptimizationParams, get_combined_args, ModelHiddenParams
from scene.gaussian_model_nogrid import GaussianModel_nogrid as GaussianModel
from time import time
from scipy.spatial.transform import Rotation as R

def prepare_offset(rotation, translation):
    def func(pts):
        return (torch.from_numpy(rotation).float().cuda() @ pts.permute(1, 0)).permute(1, 0) + torch.from_numpy(translation).float().cuda()
    return func

def find_rotation_matrix(v1, v2):
    """
    Find the rotation matrix that aligns v1 to v2.

    Parameters:
    - v1: The initial vector.
    - v2: The target vector.

    Returns:
    - The rotation matrix that rotates v1 to align with v2.
    """
    # Normalize the target vector
    if np.linalg.norm(v2) > 1e-3:
        v2_normalized = v2 / np.linalg.norm(v2)
    else:
        v2_normalized = v2
    
    # Axis of rotation (cross product of v1 and v2)
    axis = np.cross(v1, v2_normalized)

    if np.linalg.norm(axis) < 1e-6:
        if np.dot(v1, v2) >= 0:
            # The vectors are parallel, no rotation needed
            rotation_matrix = np.eye(3)
        else:
            # The vectors are anti-parallel, rotate 180 degrees around any orthogonal axis
            rotation_matrix = R.from_euler('x', 180, degrees=True).as_matrix()

    else:
        # Angle of rotation
        angle = np.arccos(np.dot(v1, v2_normalized))

        # Handle the case where the rotation is undefined because the vectors are parallel/anti-parallel
        
        # Normalize the rotation axis
        axis = axis / np.linalg.norm(axis)
        
        # Rodrigues' rotation formula components
        K = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
        I = np.identity(3)
        
        # Rotation matrix
        rotation_matrix = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
    return rotation_matrix # [3, 3]

def get_rotation(prev_pos, next_pos):
    new_vec = next_pos - prev_pos
    canonical = np.array([1, 0, 0])
    return find_rotation_matrix(canonical, new_vec)


def query_trajectory(generate_coordinates, t0, fps, frame_num):
    # get_location = lambda t: np.array((R * np.sin(2 * np.pi * t * rot_speed), 0, R * np.cos(2 * np.pi * t * rot_speed)))
    translation_list = [generate_coordinates(t0 + i * fps) for i in range(frame_num)]
    return translation_list

# def query_trajectory(t0, fps, frame_num):
#     R = 0.5
#     rot_speed = 1 / 3
#     get_location = lambda t: np.array((R * np.sin(2 * np.pi * t * rot_speed), 0, R * np.cos(2 * np.pi * t * rot_speed)))
#     translation_list = [get_location(t0 + i * fps) for i in range(frame_num)]
#     return translation_list

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set_fixcam(model_path, name, iteration, views, gaussians, pipeline, background,multiview_video, fname='video_rgb.mp4', func=None, scales=None, pre_scale=False, cam_idx=25):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print(len(views))
    
    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    # for idx in tqdm(range (100)):
    # fnum = 100
    # fnum = 12
    ####
    fnum = 48
    for idx in tqdm(range (fnum)):
        view = views[cam_idx]
        if idx == 0:time1 = time()
        #ww = torch.tensor([idx / 12]).unsqueeze(0)
        ww = torch.tensor([idx / fnum]).unsqueeze(0)
        # ww = torch.tensor([idx / 100]).unsqueeze(0)

        # if multiview_video:
        # print(idx, len(func), view.keys(), len(scales))
        rendering = render(view['cur_cam'], gaussians, pipeline, background, time=ww, stage='fine', offset=[lambda x:x, func[idx]], scales_list=scales, pre_scale=pre_scale)["render"]
        # else:
        #     rendering = render(view['pose0_cam'], gaussians, pipeline, background, time=ww, stage='fine', offset=[lambda x:x, func[idx]], scales_list=scales, pre_scale=pre_scale)["render"]
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    print('Len', len(render_images))
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), fname), render_images, fps=8, quality=8)
    
def render_set_fixtime(model_path, name, iteration, views, gaussians, pipeline, background,multiview_video, fname='video_rgb.mp4', func=None, scales=None, pre_scale=False, time_idx=8):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print(len(views))
    
    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    # for idx in tqdm(range (100)):
    fnum = 100
    # fnum = 12
    for idx in tqdm(range (fnum)):
        view = views[idx]
        if idx == 0:time1 = time()
        #ww = torch.tensor([idx / 12]).unsqueeze(0)
        ww = torch.tensor([time_idx / fnum]).unsqueeze(0)
        # ww = torch.tensor([idx / 100]).unsqueeze(0)

        # if multiview_video:
        # print(idx,)
        rendering = render(view['cur_cam'], gaussians, pipeline, background, time=ww, stage='fine', offset=[lambda x:x, func[time_idx]], scales_list=scales, pre_scale=pre_scale)["render"]
        # else:
        #     rendering = render(view['pose0_cam'], gaussians, pipeline, background, time=ww, stage='fine', offset=[lambda x:x, func[idx]], scales_list=scales, pre_scale=pre_scale)["render"]
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    print('Len', len(render_images))
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), fname), render_images, fps=8, quality=8)
    

from importlib import import_module
def render_sets(dataset : ModelParams, hyperparam, opt,iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool,multiview_video: bool):

    func_name = opt.func_name
    p, m = func_name.rsplit('.', 1)
    mod = import_module(p)
    generate_coordinates = getattr(mod, m)
    translation_list = query_trajectory(generate_coordinates, 0, 1 / 48, 48 + 1)
    print('translation', translation_list)
    rotation_list = [get_rotation(translation_list[i], translation_list[i + 1]) for i in range(len(translation_list) - 1)]
    print(rotation_list)
    func = [prepare_offset(rotation_list[i], translation_list[i]) for i in range(len(rotation_list))]

    with torch.no_grad():
        gaussians = [GaussianModel(dataset.sh_degree, hyperparam) for __ in dataset.cloud_path]
        # gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        offset_list = []
        for gs in scene.gaussians:
            offset_list.append(lambda x:x)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not skip_video:
            #origin
            for cam_idx in range(0, 100, 5):
                render_set_fixcam(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,multiview_video=False, fname=f"pose_{cam_idx}.mp4", func=func, scales=opt.scales, pre_scale=opt.pre_scale, cam_idx=cam_idx)
            # for time in range(48):
            #     render_set_fixtime(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,multiview_video=False, fname=f"time_{time}.mp4", func=func, scales=opt.scales, pre_scale=opt.pre_scale, time_idx=time)
            # render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,multiview_video=True, fname='multiview.mp4', func=func, scales=opt.scales, pre_scale=opt.pre_scale)
        # self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument('--multiview_video',default=False,action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        # import mmcv
        import mmengine
        from utils.params_utils import merge_hparams
        # config = mmcv.Config.fromfile(args.configs)
        config = mmengine.Config.fromfile(args.configs)
        # import mmcv
        # from utils.params_utils import merge_hparams
        # config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), op.extract(args),args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video,args.multiview_video)
