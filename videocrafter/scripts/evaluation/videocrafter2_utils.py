from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
import sys
import argparse, os
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import torchvision, tqdm
# from utils.utils import instantiate_from_config
# from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
# from funcs import batch_ddim_sampling

# from lvdm.models.samplers.ddim import DDIMSampler
import importlib
from collections import OrderedDict




def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu")
        try:
            ## deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=full_strict)
        return model
    load_checkpoint(model, ckpt, full_strict=True)
    print('>>> model checkpoint loaded.')
    return model

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """ 
    name: full name of source para
    para_list: partial name of target para 
    """
    istarget=False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=str, default=16, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    # parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    # parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--cfg", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--cfg_temporal", type=float, default=0.0, help="prompt classifier-free guidance")
    # parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    ## for conditional i2v only
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")

    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument("--vram_O", action="store_true", help="optimization for low VRAM usage")
    parser.add_argument("--use_rgb", action="store_true", help="use rgb")

    return parser

class VideoCrafter2(nn.Module):
    def __init__(
        self,
        model_config,
        ckpt_path,
        device=torch.device("cuda"),
        weights_dtype=torch.float32
    ):
        super().__init__()

        self.model = instantiate_from_config(model_config).to(weights_dtype)
        self.device = device
        self.weights_dtype = weights_dtype
        if ckpt_path:
            self.model = load_model_checkpoint(self.model, ckpt_path).to(weights_dtype)
        self.model.model.diffusion_model.dtype = weights_dtype
        self.model.to(device)
        self.model.eval()
        print(f"{self.model.parameterization} {self.model.dtype} {self.model.model.diffusion_model.dtype}")
        self._init_train()

    def _init_train(self, t_range=[0.02, 0.98]):
        total_steps = self.model.num_timesteps
        self.min_step = int(total_steps * t_range[0])
        self.max_step = int(total_steps * t_range[1])
        self.alphas = self.model.alphas_cumprod.to(self.weights_dtype).to(self.device)
        self.sigmas = ((1 - self.model.alphas_cumprod) ** 0.5).to(self.weights_dtype).to(self.device)
       
    def train_step(self, rgbs, cond, un_cond, cfg=10.0, cfg_temporal=0.0, as_latent=False):
        batch_size = rgbs.shape[0]
        t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)
        rgbs = rgbs.to(self.weights_dtype)
        if as_latent:
            rgbs_latent = rgbs
        else:
            rgbs_latent = self.model.encode_first_stage(rgbs)
        with torch.no_grad():
            noise = torch.randn_like(rgbs_latent).to(self.weights_dtype)
            rgb_noisy = self.model.q_sample(x_start=rgbs_latent, t=t, noise=noise).to(self.weights_dtype)
            noise_pred_text= self.model.apply_model(rgb_noisy, t.to(self.weights_dtype), cond)
            noise_pred_uncond= self.model.apply_model(rgb_noisy, t.to(self.weights_dtype), un_cond)

            if cfg_temporal:
                noise_pred_static = self.model.apply_model(rgb_noisy, t.to(self.weights_dtype), cond, no_temporal_attn=True).to(self.weights_dtype)

        # perform guidance (high scale from paper!)
        noise_pred_cond = noise_pred_text + cfg * (
            noise_pred_text - noise_pred_uncond
        )
        if cfg_temporal:
            noise_pred_cond += cfg_temporal * (noise_pred_text - noise_pred_static)

        weight = (1 - self.alphas[t]).view(-1, 1, 1, 1, 1)

        # print(noise_pred_cond.shape, noise.shape)
        grad = weight * (noise_pred_cond - noise)
        target = (rgbs_latent - grad).detach()
        loss_sds = 0.5 * F.mse_loss(rgbs_latent.float(), target, reduction='sum') / rgbs_latent.shape[0]
        # print(f"loss_sds {loss_sds}")

        # latents_1step_orig = (
        #     1
        #     / self.alphas[t].view(-1, 1, 1, 1)
        #     * (rgb_noisy - self.sigmas[t].view(-1, 1, 1, 1) * noise_pred_cond)
        # ).detach()
        # with torch.no_grad():
        #     # rgb_target = self.model.decode_first_stage(target.to(self.weights_dtype))
        #     image_denoised_pretrain = self.model.decode_first_stage(latents_1step_orig)
        # grad_img = (
        #         weight
        #         * (rgbs - image_denoised_pretrain)
        #         * self.alphas[t].view(-1, 1, 1, 1)
        #         / self.sigmas[t].view(-1, 1, 1, 1)
        #     )
        # target_img = (rgbs - grad_img).detach()
        # recon_loss = F.mse_loss(rgbs.float(), target_img.detach().float(), reduction="sum") / target_img.shape[0]
        # loss_sds += 0.01 * torch.nan_to_num(recon_loss)
        # print(f"recon_loss {recon_loss}")

        return loss_sds
    
    def decode_latent(self, rgbs_latent):
        return self.model.decode_first_stage(rgbs_latent)
    
if __name__ == "__main__":
    @torch.no_grad()
    def save_results(results, filename, fps=10):
        # print('results.shape :', results.shape)
        video = results.permute(2, 0, 1, 3, 4) # [t, sample_num, c, h, w]
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(video.shape[1])) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        # already in [0,1]
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        # torchvision.io.write_video(filename, grid, fps=fps, video_codec='h264', options={'crf': '10'})
        imageio.mimwrite(filename, grid, format='gif')
        # imageio.mimwrite(filename, grid, format='mp4', fps=8)


    parser = get_parser()
    opt = parser.parse_args()
    seed_everything(opt.seed)
    device = torch.device("cuda")
    
    weights_dtype = torch.float16 if opt.fp16 else torch.float32

    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(opt.config)
    model_config = config.pop("model", OmegaConf.create())
    # model = instantiate_from_config(model_config)
    vc2 = VideoCrafter2(model_config, ckpt_path=opt.ckpt_path, weights_dtype=weights_dtype, device=device)
    ## saving folders
    os.makedirs(opt.savedir, exist_ok=True)

    ## step 2: load data
    ## -----------------------------------------------------------------
    assert os.path.exists(opt.prompt_file), "Error: prompt file NOT Found!"
    prompt_list = load_prompts(opt.prompt_file)
    num_samples = len(prompt_list)
    filename_list = [f"{id+1:04d}" for id in range(num_samples)]

    ## step 3: run over samples
    ## -----------------------------------------------------------------
    for prompts, filename in zip(prompt_list, filename_list):
        if isinstance(prompts, str):
            prompts = [prompts]
        with torch.no_grad():
            text_emb = vc2.model.get_learned_conditioning(prompts)
            neg_prompt_emb = vc2.model.get_learned_conditioning(["text, watermark, copyright, blurry, nsfw"])

        ## sample shape
        assert (opt.height % 16 == 0) and (opt.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        frames = vc2.model.temporal_length if opt.frames < 0 else opt.frames
        batch_size = 1
        fps = torch.tensor([opt.fps]*batch_size).to(vc2.model.device).long()

        if opt.use_rgb:
            # rgbs = torch.ones(batch_size, 3, frames, opt.height, opt.width).to(weights_dtype).to(device)
            rgbs = torch.randn(batch_size, 3, 1, opt.height, opt.width).repeat(1, 1, frames, 1, 1).clamp(0, 1).to(weights_dtype).to(device) # works better
            rgbs.requires_grad = True

            optimizer = torch.optim.Adam([rgbs], lr=opt.lr)
            cond = {"c_crossattn": [text_emb], "fps": fps}
            un_cond = {"c_crossattn": [neg_prompt_emb], "fps": fps}
            for step in tqdm.tqdm(range(1001)):
                optimizer.zero_grad()
                loss_sds = vc2.train_step(rgbs, cond, un_cond, cfg=opt.cfg, cfg_temporal=opt.cfg_temporal, as_latent=False)
                loss_sds.backward()
                optimizer.step()

                if step % 100 == 0:
                    tqdm.tqdm.write(f"step: {step}, loss_sds: {loss_sds.item()}")
                    video_path = os.path.join(opt.savedir, f"{filename}_sds_{step}.gif")
                    out = rgbs.detach().float().clamp(0, 1)
                    save_results(out.data.cpu(), video_path, fps=opt.savefps)
        
        else:
            ## latent noise shape
            h, w = opt.height // 8, opt.width // 8
            latent_channels = vc2.model.channels
            rgbs_latent = torch.randn(batch_size, latent_channels, 1, h, w).repeat(1, 1, frames, 1, 1).to(device)
            rgbs_latent.requires_grad = True

            optimizer = torch.optim.Adam([rgbs_latent], lr=opt.lr)
            cond = {"c_crossattn": [text_emb], "fps": fps}
            un_cond = {"c_crossattn": [neg_prompt_emb], "fps": fps}
            for step in tqdm.tqdm(range(1001)):
                optimizer.zero_grad()
                loss_sds = vc2.train_step(rgbs_latent, cond, un_cond, cfg=opt.cfg, cfg_temporal=opt.cfg_temporal, as_latent=True)
                loss_sds.backward()
                optimizer.step()
                if step % 100 == 0:
                    tqdm.tqdm.write(f"step: {step}, loss_sds: {loss_sds.item()}")
                    # print(f"step: {step}, loss_sds: {loss_sds.item()}")
                with torch.no_grad():
                    if step % 100 == 0:
                        video_path = os.path.join(opt.savedir, f"{filename}_sds_{step}.gif")
                        # out = model.decode_first_stage_2DAE(rgbs_latent.detach())
                        out = vc2.decode_latent(rgbs_latent.detach())
                        out = out.float()
                        out = (out / 2 + 0.5).clamp(0, 1)
                        save_results(out.data.cpu(), video_path, fps=opt.savefps)

