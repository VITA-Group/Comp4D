from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
)
# from diffusers.utils.import_utils import is_xformers_available

from typing import List

# suppress partial model loading warning
logging.set_verbosity_error()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class ZeroScope(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        t_range=[0.2, 0.8],
        # t_range=[0.02, 0.98],
    ):
        # # sd_version="2.1",
        # hf_key=None,
        super().__init__()

        self.device = device
        # self.sd_version = sd_version
        model_key = 'cerspense/zeroscope_v2_576w'
        self.weights_dtype = torch.float16 if fp16 else torch.float32

        # if hf_key is not None:
        #     print(f"[INFO] using hugging face custom model key: {hf_key}")
        #     model_key = hf_key
        # elif self.sd_version == "2.1":
        #     model_key = "stabilityai/stable-diffusion-2-1-base"
        # elif self.sd_version == "2.0":
        #     model_key = "stabilityai/stable-diffusion-2-base"
        # elif self.sd_version == "1.5":
        #     model_key = "runwayml/stable-diffusion-v1-5"
        # else:
        #     raise ValueError(
        #         f"Stable-diffusion version {self.sd_version} not supported."
        #     )

        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype
        )

        # if vram_O:
        # pipe.enable_sequential_cpu_offload()
        # pipe.enable_vae_slicing()
        # pipe.unet.to(memory_format=torch.channels_last)
        # pipe.enable_attention_slicing(1)
        # pipe.enable_model_cpu_offload()
        # else:
        pipe.to(device)

        self.vae = pipe.vae
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.unet.eval()

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = None

    def encode_images(self, imgs, normalize: bool = True):
        # iamge is B, 3, N, 320, 576
        # breakpoint()
        if len(imgs.shape) == 4:
            print("Only given an image an not video")
            imgs = imgs[:, :, None]
        # breakpoint()
        batch_size, channels, num_frames, height, width = imgs.shape
        imgs = imgs.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        input_dtype = imgs.dtype
        if normalize:
            imgs = imgs * 2.0 - 1.0
        # breakpoint()

        # if self.cfg.low_ram_vae > 0:
        #     vnum = self.cfg.low_ram_vae
        #     mask_vae = torch.randperm(imgs.shape[0]) < vnum
        #     with torch.no_grad():
        #         posterior_mask = torch.cat(
        #             [
        #                 self.vae.encode(
        #                     imgs[~mask_vae][i : i + 1].to(self.weights_dtype)
        #                 ).latent_dist.sample()
        #                 for i in range(imgs.shape[0] - vnum)
        #             ],
        #             dim=0,
        #         )
        #     posterior = torch.cat(
        #         [
        #             self.vae.encode(
        #                 imgs[mask_vae][i : i + 1].to(self.weights_dtype)
        #             ).latent_dist.sample()
        #             for i in range(vnum)
        #         ],
        #         dim=0,
        #     )
        #     posterior_full = torch.zeros(
        #         imgs.shape[0],
        #         *posterior.shape[1:],
        #         device=posterior.device,
        #         dtype=posterior.dtype,
        #     )
        #     posterior_full[~mask_vae] = posterior_mask
        #     posterior_full[mask_vae] = posterior
        #     latents = posterior_full * self.vae.config.scaling_factor
        # else:
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        latents = (
            latents[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + latents.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        return latents.to(input_dtype)

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts=['static, low motion, static statue, not moving, no motion, text, watermark, copyright, blurry, nsfw']):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        # self.embeddings = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768] # wrong order...
        # embs = zs.encode_text(['a cat running with a dog'])
        # neg_prompt = zs.encode_text([""])
        # print(embs.shape, neg_prompt.shape)
        embeddings = torch.cat([pos_embeds, neg_embeds], dim=0)
        return embeddings
    
    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def forward_unet(
        self,
        latents,
        t,
        encoder_hidden_states,
    ):
        input_dtype = latents.dtype
        # print(latents.shape, latents.device, t.shape, t.device, encoder_hidden_states.shape, encoder_hidden_states.device)
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)



    @torch.no_grad()
    def refine(self, pred_rgb,
               guidance_scale=100, steps=50, strength=0.8,
        ):

        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=self.embeddings,
            ).sample

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs

    def train_step(
        self,
        pred_rgb,
        text_embs,
        step_ratio=None,
        guidance_scale=100,
        as_latent=False,
    ):
        # print(pred_rgb.shape)
        batch_size = pred_rgb.shape[0] // 16
        # batch_size = 1
        pred_rgb = pred_rgb.to(self.dtype) # B, C, H, W

        if as_latent:
            latents = F.interpolate(pred_rgb, (40, 72), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)[None]# * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (320, 576), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)[None]
            # encode image into latents with vae, requires grad!
            latents = self.encode_images(pred_rgb_512)
        # print(latents.shape)

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        # w(t), sigma_t^2
        # w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        # predict the noise residual with unet, NO grad!
        # with torch.no_grad():
        #     # add noise
        #     noise = torch.randn_like(latents)
        #     latents_noisy = self.scheduler.add_noise(latents, noise, t)
        #     # pred noise
        #     latent_model_input = torch.cat([latents_noisy] * 2)
        #     tt = torch.cat([t] * 2)

        #     noise_pred = self.unet(
        #         latent_model_input, tt, encoder_hidden_states=self.embeddings.repeat(batch_size, 1, 1)
        #     ).sample

        #     # perform guidance (high scale from paper!)
        #     noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + guidance_scale * (
        #         noise_pred_pos - noise_pred_uncond
        #     )
        grad = self.compute_grad_sds(latents, text_embs, t, use_csd=True).to(latents.dtype)

        # grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # seems important to avoid NaN...
        # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target.float(), reduction='sum') / latents.shape[0]
        rgb_target = self.decode_latents(target).permute(0, 2, 1, 3, 4)
        # print(latents.dtype, target.dtype, pred_rgb_512.dtype)
        # print(latents.dtype, target.dtype, pred_rgb_512.dtype, rgb_target.dtype)
        # print(pred_rgb_512.shape, rgb_target.shape)
        loss += 0.05 * F.mse_loss(pred_rgb_512.float(), rgb_target.detach().float(), reduction='sum') / rgb_target.shape[0]
        # loss += 0.05 * F.mse_loss(pred_rgb_512, rgb_target.half().detach(), reduction='sum') / rgb_target.shape[0]

        return loss
        # return loss.half()

    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    self.embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=self.embeddings
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    # def decode_latents(self, latents):
    #     latents = 1 / self.vae.config.scaling_factor * latents

    #     imgs = self.vae.decode(latents).sample
    #     imgs = (imgs / 2 + 0.5).clamp(0, 1)

    #     return imgs
    @torch.no_grad()
    def decode_latents(self, latents):
        # TODO: Make decoding align with previous version
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )

        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            # .permute(0, 2, 1, 3, 4)
        )
        # video = video.permute(0, )
        # print(video.shape)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # video = video.float()
        video = (video / 2 + 0.5).clamp(0, 1)
        return video

    def compute_grad_csd(
        self,
        latents,
        text_embeddings,
        t,
    ):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = 100 * (
        # noise_pred = noise_pred_text + 100 * (
            noise_pred_text - noise_pred_uncond
        )

        # if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        # elif self.cfg.weighting_strategy == "uniform":
        #     w = 1
        # elif self.cfg.weighting_strategy == "fantasia3d":
        #     w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        # else:
        #     raise ValueError(
        #         f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
        #     )

        grad = w * (noise_pred)
        # grad = w * (noise_pred - noise)
        return grad

    def compute_grad_sds(
        self,
        latents,
        text_embeddings,
        t,
        use_csd=False
    ):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + 100 * (
            noise_pred_text - noise_pred_uncond
        )

        # if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        # elif self.cfg.weighting_strategy == "uniform":
        #     w = 1
        # elif self.cfg.weighting_strategy == "fantasia3d":
        #     w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        # else:
        #     raise ValueError(
        #         f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
        #     )
        if use_csd:
            grad = w * (noise_pred - noise_pred_text)
        else:
            grad = w * (noise_pred - noise)
        return grad


    # def encode_imgs(self, imgs):
    #     # imgs: [B, 3, H, W]

    #     imgs = 2 * imgs - 1

    #     posterior = self.vae.encode(imgs).latent_dist
    #     latents = posterior.sample() * self.vae.config.scaling_factor

    #     return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        # self.get_text_embeds(prompts, negative_prompts)
        
        # # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs

    @torch.no_grad()
    def generate_img(
        self,
        emb,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        neg_prompt = self.encode_text([""])
        self.embeddings = torch.cat([neg_prompt, emb.unsqueeze(0)], dim=0)  #
        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


def window_score(x, gamma: float = 0.6) -> torch.Tensor:
    # return torch.exp(-torch.abs(gamma*x))
    return torch.cos(gamma*x)


# Collect similar info from attentive features for neglected concept
def sim_correction(embeddings: torch.Tensor,
                   correction_indices: List[int],
                   scores: torch.Tensor,
                   window: bool = True) -> torch.Tensor:
    """ Embeddings shape (77, 768), computes similarity between embeddings, combine using similarity scores"""
    ntk, dim = embeddings.shape
    device = embeddings.device

    for i, tk in enumerate(correction_indices):
        alpha = scores[i]
        v = embeddings[tk].clone()

        sim = v.unsqueeze(0) * embeddings  # nth,dim 77,768
        sim = torch.relu(sim)  # 77,768

        ind = torch.lt(sim, 0.5)  # relu is not needed in this case
        sim[ind] = 0.
        sim[:tk] = 0.  # 77, 768
        sim /= max(sim.max(), 1e-6)

        if window:
            ws = window_score(torch.arange(0, ntk - tk).to(device), gamma=0.8)
            ws = ws.unsqueeze(-1)  # 77 - tk,1
            sim[tk:] = ws * sim[tk:]  # 77, 768

        successor = torch.sum(sim * embeddings, dim=0)
        embeddings[tk] = (1 - alpha) * embeddings[tk] + alpha * successor
        embeddings[tk] *= v.norm() / embeddings[tk].norm()

    return embeddings

if __name__ == "__main__":
    import torchvision, tqdm
    @torch.no_grad()
    def save_results(results, filename, fps=10):
        video = results.permute(1, 0, 2, 3, 4) # [t, sample_num, c, h, w]
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(video.shape[1])) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        # already in [0,1]
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(filename, grid, fps=fps, video_codec='h264', options={'crf': '10'})

    import argparse, os
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    # parser.add_argument("prompt", type=str)
    # parser.add_argument("--negative", default="", type=str)
    # parser.add_argument(
    #     "--sd_version",
    #     type=str,
    #     default="1.5",
    #     choices=["1.5", "2.0", "2.1"],
    #     help="stable diffusion version",
    # )
    # parser.add_argument(
    #     "--hf_key",
    #     type=str,
    #     default=None,
    #     help="hugging face Stable diffusion model key",
    # )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    zs = ZeroScope(device, opt.fp16, opt.vram_O)
    # sd = ZeroScope(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    # imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    # plt.imshow(imgs[0])
    # plt.show()
    embs = zs.encode_text(['a panda dancing'])
    # embs = zs.encode_text(['a bee flying around a flower'])
    # embs = zs.encode_text(['a cat running with a dog'])
    neg_prompt = zs.encode_text(['static, low motion, static statue, not moving, no motion, text, watermark, copyright, blurry, nsfw'])
    # print(embs.shape, neg_prompt.shape)
    embeddings = torch.cat([embs, neg_prompt], dim=0)  #
    # embeddings = torch.cat([neg_prompt, embs], dim=0)  #
    # embeddings = torch.cat([neg_prompt, embs.unsqueeze(0)], dim=0)  #
    # in rgb
    # use_rgb = True
    use_rgb = False
    if use_rgb:
        rgbs = torch.rand(1, 3, 320, 576).cuda().repeat(16, 1, 1, 1).clamp(0, 1)
        rgbs.requires_grad = True
        optimizer = torch.optim.Adam([rgbs], lr=0.1)
        for step in tqdm.tqdm(range(1000)):
            optimizer.zero_grad()
            loss_sds = zs.train_step(rgbs, embeddings)
            loss_sds.backward()
            optimizer.step()
            if step % 20 == 0:
                tqdm.tqdm.write(f"step: {step}, loss_sds: {loss_sds.item()}")
                # print(f"step: {step}, loss_sds: {loss_sds.item()}")
            if step % 20 == 0:
                video_path = os.path.join('./output', f"sds_rgb_{step}.mp4")
                save_results(rgbs.data.cpu().unsqueeze(0), video_path, fps=10)
    else:
        rgbs = torch.randn(1, 4, 40, 72).cuda().repeat(16, 1, 1, 1)
        rgbs.requires_grad = True
        optimizer = torch.optim.Adam([rgbs], lr=0.1)
        for step in tqdm.tqdm(range(1001)):
            optimizer.zero_grad()
            loss_sds = zs.train_step(rgbs, embeddings, as_latent=True)
            loss_sds.backward()
            if step % 20 == 0:
                tqdm.tqdm.write(f"step: {step}, loss_sds: {loss_sds.item()}")
                # print(f"step: {step}, loss_sds: {loss_sds.item()}")
            optimizer.step()
            with torch.no_grad():
                if step % 100 == 0:
                # if step % 100 == 0 and step > 0:
                    video_path = os.path.join('./output', f"sds_{step}.mp4")
                    out = zs.decode_latents(rgbs.permute(1, 0, 2, 3)[None].detach())
                    save_results(out.data.cpu(), video_path, fps=10)
    # ww = sd.encode_text('A teddy bear with a yellow bird')
    # token_indices = [5, 8]
    # cor_scores1 = [0.3, 0]
    # from IPython import embed
    # embed()
    # res = sim_correction(embeddings=ww[0], correction_indices=token_indices, scores=torch.tensor(cor_scores1, device=device))

    # imgs = sd.generate_img(res, opt.H, opt.W, opt.steps)
    # from PIL import Image
    # for i in range(len(imgs)):
    #     Image.fromarray(imgs[i]).save(f'b_{i}.png')
    # imgs = sd.generate_img(ww[0], opt.H, opt.W, opt.steps)
    # from PIL import Image
    # for i in range(len(imgs)):
    #     Image.fromarray(imgs[i]).save(f'c_{i}.png')
