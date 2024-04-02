import numpy as np
import random
import os
import torch

from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer.comp_renderer import render as render_comp
from gaussian_renderer import render as render_single
import sys
from scene.comp_scene import Scene
from scene.gaussian_model_nogrid import GaussianModel_nogrid as GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from importlib import import_module

# import lpips
import gc
from torchvision import transforms as T
from utils.scene_utils import render_training_image
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# from guidance.zero123_utils import Zero123
# from guidance.zeroscope_utils_hifa import ZeroScope
# from guidance.zeroscope_utils import ZeroScope
# from guidance.mvdream_utils import MVDream
from guidance.sd_utils import StableDiffusion

from PIL import Image
from torchvision.transforms import ToTensor
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R

def prepare_offset(rotation, translation):
    def func(pts):
        return (torch.from_numpy(rotation).float().cuda().detach() @ pts.permute(1, 0)).permute(1, 0) + torch.from_numpy(translation).float().cuda().detach()
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
    # canonical = np.array([0, 0, 1])
    return find_rotation_matrix(canonical, new_vec)

# Constants
g = 9.81  # acceleration due to gravity, m/s^2

# Initial horizontal velocity calculation
vx = 2  # m/s, to cover 4 meters in 2 seconds

# To calculate the initial vertical velocity, we use the equation of motion at the peak (1 second into the jump)
# At the peak, vertical velocity (v) = 0, acceleration (a) = -g, time (t) = 1 sec
# We rearrange the equation v = u + at to find u: u = v - at
vy_initial = 0 - (-g) * 1  # Initial vertical velocity

    # return np.array((x, z, y))
    # return np.array((x, y, z))

def query_trajectory(generate_coordinates, t0, fps, frame_num):
    # get_location = lambda t: np.array((R * np.sin(2 * np.pi * t * rot_speed), 0, R * np.cos(2 * np.pi * t * rot_speed)))
    translation_list = [generate_coordinates(t0 + i * fps) for i in range(frame_num)]
    return translation_list

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer, args):
    first_iter = 0

    torch.cuda.empty_cache()
    gc.collect()
    print(f'Start training of stage {stage}: ')
    obj_prompts = []
    if opt.video_sds_type == 'zeroscope':
        from guidance.zeroscope_utils import ZeroScope
        zeroscope = ZeroScope('cuda', fp16=True)
        emb_zs = zeroscope.get_text_embeds([opt.prompt])
        for ww in opt.obj_prompt:
            obj_prompts.append(zeroscope.get_text_embeds([ww]))
    else:
        from videocrafter.scripts.evaluation.videocrafter2_utils import VideoCrafter2
        from omegaconf import OmegaConf
        vc_model_config = OmegaConf.load('videocrafter/configs/inference_t2v_512_v2.0.yaml').pop("model", OmegaConf.create())
        vc2 = VideoCrafter2(vc_model_config, ckpt_path='model.ckpt', weights_dtype=torch.float16, device='cuda')
        emb_zs = vc2.model.get_learned_conditioning([opt.prompt])
        neg_emb_zs = vc2.model.get_learned_conditioning(["text, watermark, copyright, blurry, nsfw"])
        cond = {"c_crossattn": [emb_zs], "fps": torch.tensor([6]*emb_zs.shape[0]).to(vc2.model.device).long()}
        un_cond = {"c_crossattn": [neg_emb_zs], "fps": torch.tensor([6]*emb_zs.shape[0]).to(vc2.model.device).long()}
        
        for ww in opt.obj_prompt:
            emb_zs = vc2.model.get_learned_conditioning([ww])
            obj_prompts.append({"c_crossattn": [emb_zs], "fps": torch.tensor([6]*emb_zs.shape[0]).to(vc2.model.device).long()})

    sd = StableDiffusion('cuda', fp16=True, sd_version='2.1')
    sd.get_text_embeds([opt.prompt], negative_prompts=['static statue, text, watermark, copyright, blurry, nsfw'])
    sd.get_objects_text_embeds(opt.obj_prompt, negative_prompts=['static statue, text, watermark, copyright, blurry, nsfw'])
    
    stage_ = ['fine']
    train_iter_ = [opt.iterations]
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda", requires_grad=False)
    black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda", requires_grad=False)

    for cur_stage, train_iter in zip(stage_, train_iter_):
        for gs in gaussians:
            gs.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            for gs in gaussians:
                gs.restore(model_params, opt)
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        viewpoint_stack = None
        ema_loss_for_log = 0.0

        final_iter = train_iter

        progress_bar = tqdm(range(first_iter, final_iter), desc=f"[{args.expname}] Training progress")
        offset_list = []
        for gs in gaussians:
            offset_list.append(lambda x:x)

        func_name = opt.func_name
        p, m = func_name.rsplit('.', 1)
        mod = import_module(p)
        generate_coordinates = getattr(mod, m)

        translation_list = query_trajectory(generate_coordinates, 0, 1 / 16, 16 + 1)
        rotation_list = [get_rotation(translation_list[i], translation_list[i + 1]) for i in range(len(translation_list) - 1)]
        func = [prepare_offset(rotation_list[i], translation_list[i]) for i in range(len(rotation_list))]

        for iteration in range(first_iter, final_iter+1):
            stage = cur_stage
            loss_weight = 1
            if np.random.random() < 0.5:
                background = white_bg
            else:
                background = black_bg

            iter_start.record()
            for gs in gaussians:
                gs.update_learning_rate(iteration)
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras()
                viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=1,shuffle=True,num_workers=4,collate_fn=list)
                frame_num = viewpoint_stack.pose0_num

                loader = iter(viewpoint_stack_loader)
            if True:
                try:
                    data = next(loader)
                except StopIteration:
                    print("reset dataloader")
                    batch_size = 1
                    loader = iter(viewpoint_stack_loader)
            if (iteration - 1) == debug_from:
                pipe.debug = True
            images = []
            radii_list = []
            visibility_filter_list = []
            viewspace_point_tensor_list = []
            dx = []
            out_pts = []
            viewpoint_cam = data[0]['rand_poses']
            fps = 1 / frame_num
            t0 = 0
            sds_idx_list = range(frame_num)

            if np.random.random() < 0.8:
                use_comp = True
            else:
                use_comp = False
            for i in sds_idx_list:
                time = torch.tensor([t0 + i * fps]).unsqueeze(0).float()
                offset_list[-1] = func[i]
                if use_comp:
                    render_pkg = render_comp(viewpoint_cam[0], gaussians, pipe, background, stage=stage, time=time, offset=offset_list, scales_list=opt.scales, pre_scale=opt.pre_scale)
                else:
                    # render individual object
                    gs_idx = random.choice(range(len(gaussians)))
                    render_pkg = render_single(viewpoint_cam[0], (gaussians[gs_idx]), pipe, background, stage=stage, time=time, offset=offset_list[gs_idx], scales_preset=opt.scales[gs_idx], pre_scale=opt.pre_scale)

                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                fg_mask = render_pkg['alpha']
                rgba = torch.cat([image, fg_mask], dim=0)
                images.append(rgba.unsqueeze(0))
                if 'dx' in render_pkg:
                    dx.append(render_pkg['dx'])
                radii_list.append(radii.unsqueeze(0))
                visibility_filter_list.append(visibility_filter.unsqueeze(0))
                viewspace_point_tensor_list.append(viewspace_point_tensor)
            radii = torch.cat(radii_list,0).max(dim=0).values
            visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
            image_tensor = torch.cat(images,0)
            # print('output', image_tensor.shape) # B, C, H, W
            if len(out_pts):
                out_pts = torch.stack(out_pts, 0)

            if use_comp:
                if opt.video_sds_type == 'zeroscope':
                    loss = zeroscope.train_step(image_tensor[:, :3], emb_zs)
                else:
                    loss = vc2.train_step(image_tensor[:, :3].unsqueeze(0).permute(0, 2, 1, 3, 4), cond, un_cond, cfg=opt.cfg, cfg_temporal=opt.cfg_temporal, as_latent=False)

                # img loss for comp renderings
                randints = list(range(16))
                np.random.shuffle(randints)
                img_loss = sd.train_step(image_tensor[randints[0]:randints[0]+1, :3], background=background) + sd.train_step(image_tensor[randints[1]:randints[1]+1, :3], background=background) \
                    + sd.train_step(image_tensor[randints[2]:randints[2]+1, :3], background=background) + sd.train_step(image_tensor[randints[3]:randints[3]+1, :3], background=background)
                print(f"origin loss is {loss}, image_loss with weight {opt.image_weight} is {img_loss * opt.image_weight}")
                loss = img_loss * opt.image_weight + loss * loss_weight

                if opt.with_reg:
                    dx_nn_loss = []
                    for cur_dx in dx:
                        tot = cur_dx.shape[0]
                        dx_nn_loss.append(gaussians[0].get_nn_loss(cur_dx[:tot//2]))
                        dx_nn_loss.append(gaussians[1].get_nn_loss(cur_dx[tot//2:]))

                    # values inside the list are already mean-ed
                    loss_nn = torch.stack(dx_nn_loss).sum()
                    tb_writer.add_scalar(f'{stage}/dx_nn_comp', loss_nn.item(), iteration)
                    print(f'in comp loss_nn with weight {opt.nn_weight} is {loss_nn * opt.nn_weight}')
                    loss += loss_nn * opt.nn_weight
            else:
                # print(len(obj_prompts), gs_idx)
                if opt.video_sds_type == 'zeroscope':
                    loss = zeroscope.train_step(image_tensor[:, :3], obj_prompts[gs_idx])
                else:
                    loss = vc2.train_step(image_tensor[:, :3].unsqueeze(0).permute(0, 2, 1, 3, 4), obj_prompts[gs_idx], un_cond, cfg=opt.cfg, cfg_temporal=opt.cfg_temporal, as_latent=False)
                
                randints = list(range(16))
                np.random.shuffle(randints)
                img_loss = sd.train_step(image_tensor[randints[0]:randints[0]+1, :3], background=background, obj_id=gs_idx) + sd.train_step(image_tensor[randints[1]:randints[1]+1, :3], background=background, obj_id=gs_idx) \
                    + sd.train_step(image_tensor[randints[2]:randints[2]+1, :3], background=background, obj_id=gs_idx) + sd.train_step(image_tensor[randints[3]:randints[3]+1, :3], background=background, obj_id=gs_idx)
                print(f"origin loss is {loss}, image_loss with weight {opt.image_weight} is {img_loss * opt.image_weight}")
                loss = img_loss * opt.image_weight + loss * loss_weight

                if opt.with_reg:
                    dx_nn_loss = []
                    for cur_dx in dx:
                        dx_nn_loss.append(gaussians[gs_idx].get_nn_loss(cur_dx))
                    loss_nn = torch.stack(dx_nn_loss).sum()
                    tb_writer.add_scalar(f'{stage}/dx_nn_sep', loss_nn.item(), iteration)
                    print(f'in seperate loss_nn with weight {opt.nn_weight} is {loss_nn * opt.nn_weight}')
                    loss += loss_nn * opt.nn_weight
        
            if stage == 'fine':
                if (not use_comp) and gs_idx == 0:
                    loss_dx0 = torch.stack(dx).mean().abs()
                    tb_writer.add_scalar(f'{stage}/loss_dx0_mean', loss_dx0.item(), iteration)
                    loss_dx0 = torch.stack(dx).abs().sum()
                    loss += loss_dx0 * opt.loss_dx_weight
                    tb_writer.add_scalar(f'{stage}/loss_dx-first', loss_dx0.item(), iteration)
                else:
                    loss_dx0 = torch.stack(dx)
                    loss_dx0 = loss_dx0[:, :int(gaussians[0]._xyz.shape[0])]
                    loss_dx0 = torch.stack(dx).abs().sum()
                    loss += loss_dx0 * opt.loss_dx_weight
                    tb_writer.add_scalar(f'{stage}/loss_dx-first', loss_dx0.item(), iteration)

            if stage == "fine" and hyper.time_smoothness_weight != 0:
                tv_loss = torch.sum([gs.compute_regulation(hyper.time_smoothness_weight, hyper.plane_tv_weight, hyper.l1_time_planes) for gs in gaussians])
                loss += tv_loss
                tb_writer.add_scalar(f'{stage}/loss_tv', tv_loss.item(), iteration)
            loss.backward()
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
            for idx in range(0, len(viewspace_point_tensor_list)):
                if viewspace_point_tensor_list[idx].grad is not None:
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            iter_end.record()
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

                total_point = sum([gs._xyz.shape[0] for gs in gaussians])
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                            "point":f"{total_point}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                timer.pause()
                training_report(tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_comp, pipe, background, stage, func, scales=opt.scales)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration, stage)
                timer.start()

                if iteration < opt.iterations:
                    for gs in gaussians:
                        gs.optimizer.step()
                        gs.optimizer.zero_grad(set_to_none = True)

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, args):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = [GaussianModel(dataset.sh_degree, hyper) for __ in dataset.cloud_path] # init one GS model for each ply (object)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians,load_coarse=None)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer, args)

from datetime import datetime

def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = str(datetime.today().strftime('%Y-%m-%d')) + '/' + expname + '_' + datetime.today().strftime('%H:%M:%S')
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, pipe, bg, stage, func, scales):
    if tb_writer:
        # tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
    ww = iteration if stage == 'static' else iteration
    offset_list = []
    for gs in scene.gaussians:
        offset_list.append(lambda x:x)

    if iteration % 100 == 0 and ww in testing_iterations:
    # if stage == 'fine':
    # if ww in testing_iterations:
        torch.cuda.empty_cache()
        train_set = scene.getTrainCameras()
        validation_configs = [{'name': 'train', 'cameras' : [train_set[idx % len(train_set)] for idx in range(10, 5000, 299)]}]
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ti = (torch.tensor([0]).unsqueeze(0))
                cam_li = config['cameras'][0]['rand_poses']
                im_li = []
                num = len(cam_li)
                for tii in range(num):
                    offset_list[-1] = func[tii]
                    if stage == 'static':
                        ti = (torch.tensor([tii * 0]).unsqueeze(0).cuda())
                    else:
                        ti = (torch.tensor([tii / num]).unsqueeze(0).cuda())
                    viewpoint = cam_li[tii]
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, pipe=pipe, bg_color=bg, time=ti, offset=offset_list, scales_list=scales)["render"], 0.0, 1.0)
                    im_li.append(image)
                ww = len(im_li) // 2
                r1 = torch.cat(im_li[:ww], dim=-1)
                r2 = torch.cat(im_li[ww:], dim=-1)
                im_li = torch.cat([r1, r2], dim=-2)
                if tb_writer:
                    tb_writer.add_image(f"rand_seq/{stage}", im_li, global_step=iteration)
                l1_test = 0.0
                psnr_test = 0.0
                ti = (torch.tensor([0]).unsqueeze(0))
                cam_li = config['cameras'][0]['rand_poses']
                im_li = []
                num = len(cam_li)
                for tii in range(num):
                    offset_list[-1] = func[tii]
                    if stage == 'static':
                        ti = (torch.tensor([tii * 0]).unsqueeze(0).cuda())
                    else:
                        ti = (torch.tensor([tii / num]).unsqueeze(0).cuda())
                    viewpoint = cam_li[0]
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, pipe=pipe, bg_color=bg, time=ti, offset=offset_list, scales_list=scales)["render"], 0.0, 1.0)
                    im_li.append(image)
                ww = len(im_li) // 2
                r1 = torch.cat(im_li[:ww], dim=-1)
                r2 = torch.cat(im_li[ww:], dim=-1)
                im_li = torch.cat([r1, r2], dim=-2)
                if tb_writer:
                    tb_writer.add_image(f"static_seq/{stage}", im_li, global_step=iteration)
                print("\n[ITER {}] Evaluating {}".format(iteration, config['name']))
        if tb_writer:
            tb_writer.add_scalar(f'{stage}/total_points', scene.get_total_points(), iteration)
        torch.cuda.empty_cache()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*50 for i in range(0,300)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2500, 3000, 3500, 4000, 4500, 5000, 7000, 8000, 9000, 14000, 20000, 30_000,45000,60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('-e', "--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "arguments/comp.py")
    parser.add_argument("--yyypath", type=str, default = "")
    parser.add_argument("--t0_frame0_rate", type=float, default = 1)
    parser.add_argument("--name_override", type=str, default="")
    parser.add_argument("--sds_ratio_override", type=float, default=-1)
    parser.add_argument("--sds_weight_override", type=float, default=-1)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--image_weight_override', type=float, default=-1)
    parser.add_argument('--nn_weight_override', type=float, default=-1)
    parser.add_argument('--cfg_override', type=float, default=-1)
    parser.add_argument('--cfg_temporal_override', type=float, default=-1) 
    parser.add_argument('--loss_dx_weight_override', type=float, default=-1)
    parser.add_argument('--with_reg_override', action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations - 1)
    if args.configs:
        # import mmcv
        import mmengine
        from utils.params_utils import merge_hparams
        # config = mmcv.Config.fromfile(args.configs)
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    if args.name_override != '':
        args.name = args.name_override
    if args.sds_ratio_override != -1:
        args.fine_rand_rate = args.sds_ratio_override
    if args.sds_weight_override != -1:
        args.lambda_zero123 = args.sds_weight_override
    if args.image_weight_override != -1:
        args.image_weight = args.image_weight_override
    if args.nn_weight_override != -1:
        args.nn_weight = args.nn_weight_override
    if args.cfg_override != -1:
        args.cfg = args.cfg_override
    if args.cfg_temporal_override != -1:
        args.cfg_temporal = args.cfg_temporal_override
    if args.loss_dx_weight_override != -1:
        args.loss_dx_weight = args.loss_dx_weight_override
    if args.with_reg_override:
        args.with_reg = args.with_reg_override

    # print(args.name)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    timer1 = Timer()
    timer1.start()
    print('Configs: ', args)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args)
    print("\nTraining complete.")
    print('training time:',timer1.get_elapsed_time())
    from render_comp import render_sets

    render_sets(lp.extract(args), hp.extract(args), op.extract(args), args.iterations, pp.extract(args), skip_train=True, skip_test=True, skip_video=False, multiview_video=True)
    print("\Rendering complete.")
    