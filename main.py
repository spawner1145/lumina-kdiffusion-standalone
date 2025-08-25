import logging
import argparse
import math
import os
import random
import time
from typing import Callable, Dict, List, Optional, Tuple, Any, Union, Generator
from dataclasses import dataclass
import sys
import struct
import json

import einops
from einops import rearrange
import numpy as np
import torch
from torch import Tensor, nn
from accelerate import Accelerator
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from accelerate import init_empty_weights

import lumina_models
import strategy_lumina
from ori_schedulers import FlowMatchEulerDiscreteScheduler
from transformers import Gemma2Config, Gemma2Model
import k_diffusion_adapter

import lora_lumina

import functools
import gc

import torch
try:
    # intel gpu support for pytorch older than 2.5
    # ipex is not needed after pytorch 2.5
    import intel_extension_for_pytorch as ipex  # noqa
except Exception:
    pass

def clean_memory():
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
    if HAS_XPU:
        torch.xpu.empty_cache()
    if HAS_MPS:
        torch.mps.empty_cache()

@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float
    
def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)
    
class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h

try:
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

try:
    HAS_MPS = torch.backends.mps.is_available()
except Exception:
    HAS_MPS = False

try:
    HAS_XPU = torch.xpu.is_available()
except Exception:
    HAS_XPU = False


def denoise_k_diffusion(
    model: lumina_models.NextDiT,
    img: Tensor,
    txt: Tensor,
    txt_mask: Tensor,
    neg_txt: Tensor,
    neg_txt_mask: Tensor,
    steps: int,
    guidance_scale: float = 4.0,
    cfg_trunc_ratio: float = 1,
    renorm_cfg: float = 1.0,
    sampler: str = "euler",
    scheduler_func: str = "normal_scheduler",
    shift: float = 6.0,
    **sampler_kwargs
):
    """
    使用 k_diffusion 采样函数进行去噪
    """
    return k_diffusion_adapter.sample_with_k_diffusion(
        model=model,
        text_encoder_hidden_states=txt,
        text_encoder_attention_mask=txt_mask,
        neg_hidden_states=neg_txt,
        neg_attention_mask=neg_txt_mask,
        latents=img,
        steps=steps,
        guidance_scale=guidance_scale,
        cfg_trunc_ratio=cfg_trunc_ratio,
        renorm_cfg=renorm_cfg,
        sampler=sampler,
        scheduler_func=scheduler_func,
        shift=shift,
        **sampler_kwargs
    )
    
    
def denoise(
    scheduler,
    model: lumina_models.NextDiT,
    img: Tensor,
    txt: Tensor,
    txt_mask: Tensor,
    neg_txt: Tensor,
    neg_txt_mask: Tensor,
    timesteps: Union[List[float], torch.Tensor],
    guidance_scale: float = 4.0,
    cfg_trunc_ratio: float = 1,
    renorm_cfg: float = 1.0,
):
    for i, t in enumerate(tqdm(timesteps)):
        model.prepare_block_swap_before_forward()

        # reverse the timestep since Lumina uses t=0 as the noise and t=1 as the image
        current_timestep = 1 - t / scheduler.config.num_train_timesteps
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        current_timestep = current_timestep * torch.ones(
            img.shape[0], device=img.device
        )

        noise_pred_cond = model(
            img,
            current_timestep,
            cap_feats=txt,  # Gemma2的hidden states作为caption features
            cap_mask=txt_mask.to(dtype=torch.int32),  # Gemma2的attention mask
        )

        # compute whether to apply classifier-free guidance based on current timestep
        if current_timestep[0] < cfg_trunc_ratio:
            model.prepare_block_swap_before_forward()
            noise_pred_uncond = model(
                img,
                current_timestep,
                cap_feats=neg_txt,  # Gemma2的hidden states作为caption features
                cap_mask=neg_txt_mask.to(dtype=torch.int32),  # Gemma2的attention mask
            )
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            # apply normalization after classifier-free guidance
            if float(renorm_cfg) > 0.0:
                cond_norm = torch.linalg.vector_norm(
                    noise_pred_cond,
                    dim=tuple(range(1, len(noise_pred_cond.shape))),
                    keepdim=True,
                )
                max_new_norms = cond_norm * float(renorm_cfg)
                noise_norms = torch.linalg.vector_norm(
                    noise_pred, dim=tuple(range(1, len(noise_pred.shape))), keepdim=True
                )
                # Iterate through batch
                for i, (noise_norm, max_new_norm) in enumerate(zip(noise_norms, max_new_norms)):
                    if noise_norm >= max_new_norm:
                        noise_pred[i] = noise_pred[i] * (max_new_norm / noise_norm)
        else:
            noise_pred = noise_pred_cond

        img_dtype = img.dtype

        if img.dtype != img_dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                img = img.to(img_dtype)

        # compute the previous noisy sample x_t -> x_t-1
        noise_pred = -noise_pred
        img = scheduler.step(noise_pred, t, img, return_dict=False)[0]

    model.prepare_block_swap_before_forward()
    return img
    
class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))
    
class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h
    
class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean
    
class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))
    
def init_ipex():
    """
    Apply IPEX to CUDA hijacks using `library.ipex.ipex_init`.

    This function should run right after importing torch and before doing anything else.

    If xpu is not available, this function does nothing.
    """
    try:
        if HAS_XPU:
            from library.ipex import ipex_init

            is_initialized, error_message = ipex_init()
            if not is_initialized:
                print("failed to initialize ipex:", error_message)
        else:
            return
    except Exception as e:
        print("failed to initialize ipex:", e)

def setup_logging(args=None, log_level=None, reset=False):
    if logging.root.handlers:
        if reset:
            # remove all handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
        else:
            return

    # log_level can be set by the caller or by the args, the caller has priority. If not set, use INFO
    if log_level is None and args is not None:
        log_level = args.console_log_level
    if log_level is None:
        log_level = "INFO"
    log_level = getattr(logging, log_level)

    msg_init = None
    if args is not None and args.console_log_file:
        handler = logging.FileHandler(args.console_log_file, mode="w")
    else:
        handler = None
        if not args or not args.console_log_simple:
            try:
                from rich.logging import RichHandler
                from rich.console import Console
                from rich.logging import RichHandler

                handler = RichHandler(console=Console(stderr=True))
            except ImportError:
                # print("rich is not installed, using basic logging")
                msg_init = "rich is not installed, using basic logging"

        if handler is None:
            handler = logging.StreamHandler(sys.stdout)  # same as print
            handler.propagate = False

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)

    if msg_init is not None:
        logger = logging.getLogger(__name__)
        logger.info(msg_init)
        
def str_to_dtype(s: Optional[str], default_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    if s is None:
        return default_dtype
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif s in ["fp16", "float16"]:
        return torch.float16
    elif s in ["fp32", "float32", "float"]:
        return torch.float32
    elif s in ["fp8_e4m3fn", "e4m3fn", "float8_e4m3fn"]:
        return torch.float8_e4m3fn
    elif s in ["fp8_e4m3fnuz", "e4m3fnuz", "float8_e4m3fnuz"]:
        return torch.float8_e4m3fnuz
    elif s in ["fp8_e5m2", "e5m2", "float8_e5m2"]:
        return torch.float8_e5m2
    elif s in ["fp8_e5m2fnuz", "e5m2fnuz", "float8_e5m2fnuz"]:
        return torch.float8_e5m2fnuz
    elif s in ["fp8", "float8"]:
        return torch.float8_e4m3fn  # default fp8
    else:
        raise ValueError(f"Unsupported dtype: {s}")

init_ipex()
setup_logging()
logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=None)
def get_preferred_device() -> torch.device:
    r"""
    Do not call this function from training scripts. Use accelerator.device instead.
    """
    if HAS_CUDA:
        device = torch.device("cuda")
    elif HAS_XPU:
        device = torch.device("xpu")
    elif HAS_MPS:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"get_preferred_device() -> {device}")
    return device

def generate_image(
    model: lumina_models.NextDiT,
    gemma2: Gemma2Model,
    ae: AutoEncoder,
    prompt: str,
    system_prompt: str,
    seed: Optional[int],
    image_width: int,
    image_height: int,
    steps: int,
    guidance_scale: float,
    negative_prompt: Optional[str],
    args: argparse.Namespace,
    cfg_trunc_ratio: float = 0.25,
    renorm_cfg: float = 1.0,
):
    #
    # 0. Prepare arguments
    #
    device = get_preferred_device()
    if args.device:
        device = torch.device(args.device)

    dtype = str_to_dtype(args.dtype)
    ae_dtype = str_to_dtype(args.ae_dtype)
    gemma2_dtype = str_to_dtype(args.gemma2_dtype)

    #
    # 1. Prepare models
    #
    # model.to(device, dtype=dtype)
    model.to(device, dtype=dtype)
    model.eval()

    gemma2.to(device, dtype=gemma2_dtype)
    gemma2.eval()

    ae.to(device, dtype=ae_dtype)
    ae.eval()

    #
    # 2. Encode prompts
    #
    logger.info("Encoding prompts...")

    tokenize_strategy = strategy_lumina.LuminaTokenizeStrategy(system_prompt, args.gemma2_max_token_length)
    encoding_strategy = strategy_lumina.LuminaTextEncodingStrategy()

    tokens_and_masks = tokenize_strategy.tokenize(prompt)
    with torch.no_grad():
        gemma2_conds = encoding_strategy.encode_tokens(tokenize_strategy, [gemma2], tokens_and_masks)

    tokens_and_masks = tokenize_strategy.tokenize(
        negative_prompt, is_negative=True and not args.add_system_prompt_to_negative_prompt
    )
    with torch.no_grad():
        neg_gemma2_conds = encoding_strategy.encode_tokens(tokenize_strategy, [gemma2], tokens_and_masks)

    # Unpack Gemma2 outputs
    prompt_hidden_states, _, prompt_attention_mask = gemma2_conds
    uncond_hidden_states, _, uncond_attention_mask = neg_gemma2_conds

    if args.offload:
        print("Offloading models to CPU to save VRAM...")
        gemma2.to("cpu")
        clean_memory()

    model.to(device)

    #
    # 3. Prepare latents
    #
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    logger.info(f"Seed: {seed}")
    torch.manual_seed(seed)

    latent_height = image_height // 8
    latent_width = image_width // 8
    latent_channels = 16

    latents = torch.randn(
        (1, latent_channels, latent_height, latent_width),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    #
    # 4. Denoise
    #
    logger.info("Denoising...")
    
    with torch.autocast(device_type=device.type, dtype=dtype), torch.no_grad():
        if hasattr(args, 'use_k_diffusion') and args.use_k_diffusion:
            # 使用 k_diffusion 采样
            latents = denoise_k_diffusion(
                model,
                latents.to(device),
                prompt_hidden_states.to(device),
                prompt_attention_mask.to(device),
                uncond_hidden_states.to(device),
                uncond_attention_mask.to(device),
                steps,
                guidance_scale,
                cfg_trunc_ratio,
                renorm_cfg,
                sampler=getattr(args, 'sampler', 'euler'),
                scheduler_func=getattr(args, 'scheduler_func', 'normal_scheduler'),
                shift=args.discrete_flow_shift,
            )
        else:
            # 使用原始的调度器
            scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
            scheduler.set_timesteps(steps, device=device)
            timesteps = scheduler.timesteps
            
            latents = denoise(
                scheduler,
                model,
                latents.to(device),
                prompt_hidden_states.to(device),
                prompt_attention_mask.to(device),
                uncond_hidden_states.to(device),
                uncond_attention_mask.to(device),
                timesteps,
                guidance_scale,
                cfg_trunc_ratio,
                renorm_cfg,
            )

    if args.offload:
        model.to("cpu")
        clean_memory()
        ae.to(device)

    #
    # 5. Decode latents
    #
    logger.info("Decoding image...")
    # latents = latents / ae.scale_factor + ae.shift_factor
    with torch.no_grad():
        image = ae.decode(latents.to(ae_dtype))
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")

    #
    # 6. Save image
    #
    pil_image = Image.fromarray(image[0])
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    seed_suffix = f"_{seed}"
    output_path = os.path.join(output_dir, f"image_{ts_str}{seed_suffix}.png")
    pil_image.save(output_path)
    logger.info(f"Image saved to {output_path}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/root/autodl-tmp/Neta-Lumina/Unet/neta-lumina-v1.0.safetensors",
        help="Lumina DiT model path",
    )
    parser.add_argument(
        "--gemma2_path",
        type=str,
        default="/root/autodl-tmp/ComfyUI/models/text_encoders/gemma_2_2b_fp16.safetensors",
        help="Gemma2 model path",
    )
    parser.add_argument(
        "--ae_path",
        type=str,
        default="/root/autodl-tmp/ComfyUI/models/vae/ae.safetensors",
        help="Autoencoder model path",
    )
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the mountains", help="Prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for image generation, default is empty")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for generated images")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, default=36, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--image_width", type=int, default=1024, help="Image width")
    parser.add_argument("--image_height", type=int, default=1024, help="Image height")
    parser.add_argument("--dtype", type=str, default="bf16", help="Data type for model (bf16, fp16, float)")
    parser.add_argument("--gemma2_dtype", type=str, default="bf16", help="Data type for Gemma2 (bf16, fp16, float)")
    parser.add_argument("--ae_dtype", type=str, default="float", help="Data type for Autoencoder (bf16, fp16, float)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda:0')")
    parser.add_argument("--offload", action="store_true", help="Offload models to CPU to save VRAM")
    parser.add_argument("--system_prompt", type=str, default="", help="you are a helpful assistant to draw beautiful pictures.")
    parser.add_argument("--add_system_prompt_to_negative_prompt", action="store_true", help="Add system prompt to negative prompt")
    parser.add_argument(
        "--gemma2_max_token_length",
        type=int,
        default=256,
        help="Max token length for Gemma2 tokenizer",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=6.0,
        help="Shift value for FlowMatchEulerDiscreteScheduler",
    )
    parser.add_argument(
        "--cfg_trunc_ratio",
        type=float,
        default=0.25,
        help="The ratio of the timestep interval to apply normalization-based guidance scale. For example, 0.25 means the first 25%% of timesteps will be guided.",
    )
    parser.add_argument(
        "--renorm_cfg",
        type=float,
        default=1.0,
        help="The factor to limit the maximum norm after guidance. Default: 1.0, 0.0 means no renormalization.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use flash attention for Lumina model",
    )
    parser.add_argument(
        "--use_sage_attn",
        action="store_true",
        help="Use sage attention for Lumina model",
    )
    parser.add_argument(
        "--use_k_diffusion",
        action="store_true",
        help="Use k_diffusion sampling instead of original scheduler",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["euler", "heun", "dpm_2", "dpm_2_ancestral", "euler_ancestral"],
        help="Sampler to use when --use_k_diffusion is enabled",
    )
    parser.add_argument(
        "--scheduler_func",
        type=str,
        default="normal_scheduler",
        choices=["simple_scheduler", "ddim_scheduler", "normal_scheduler", "beta_scheduler", "linear_quadratic_schedule"],
        help="Scheduler function to use when --use_k_diffusion is enabled",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        nargs="*",
        default=[],
        help="LoRA weights, each argument is a `path;multiplier` (semi-colon separated)",
    )
    parser.add_argument("--merge_lora_weights", action="store_true", help="Merge LoRA weights to model")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode for generating multiple images / 対話モードで複数の画像を生成する",
    )
    return parser

class MemoryEfficientSafeOpen:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        return self.header.get("__metadata__", {})

    def get_tensor(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start == offset_end:
            tensor_bytes = None
        else:
            # adjust offset by header size
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)

        return self._deserialize_tensor(tensor_bytes, metadata)

    def _read_header(self):
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            tensor_bytes = bytearray(tensor_bytes)  # make it writable
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        # process float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # convert to the target dtype and reshape
        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # add float8 types if available
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            # # convert to float16 if float8 is not supported
            # print(f"Warning: {dtype_str} is not supported in this PyTorch version. Converting to float16.")
            # return byte_tensor.view(torch.uint8).to(torch.float16).reshape(shape)
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")

def load_safetensors(
    path: str, device: Union[str, torch.device], disable_mmap: bool = False, dtype: Optional[torch.dtype] = torch.float32
) -> dict[str, torch.Tensor]:
    if disable_mmap:
        # return safetensors.torch.load(open(path, "rb").read())
        # use experimental loader
        # logger.info(f"Loading without mmap (experimental)")
        state_dict = {}
        with MemoryEfficientSafeOpen(path) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key).to(device, dtype=dtype)
        return state_dict
    else:
        try:
            state_dict = load_file(path, device=device)
        except:
            state_dict = load_file(path)  # prevent device invalid Error
        if dtype is not None:
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(dtype=dtype)
        return state_dict

def load_lumina_model(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: torch.device,
    disable_mmap: bool = False,
    use_flash_attn: bool = False,
    use_sage_attn: bool = False,
):
    """
    Load the Lumina model from the checkpoint path.

    Args:
        ckpt_path (str): Path to the checkpoint.
        dtype (torch.dtype): The data type for the model.
        device (torch.device): The device to load the model on.
        disable_mmap (bool, optional): Whether to disable mmap. Defaults to False.
        use_flash_attn (bool, optional): Whether to use flash attention. Defaults to False.

    Returns:
        model (lumina_models.NextDiT): The loaded model.
    """
    logger.info("Building Lumina")
    with torch.device("meta"):
        model = lumina_models.NextDiT_2B_GQA_patch2_Adaln_Refiner(use_flash_attn=use_flash_attn, use_sage_attn=use_sage_attn).to(
            dtype
        )

    logger.info(f"Loading state dict from {ckpt_path}")
    state_dict = load_safetensors(ckpt_path, device=device, disable_mmap=disable_mmap, dtype=dtype)

    # Neta-Lumina support
    if "model.diffusion_model.cap_embedder.0.weight" in state_dict:
        # remove "model.diffusion_model." prefix
        filtered_state_dict = {
            k.replace("model.diffusion_model.", ""): v for k, v in state_dict.items() if k.startswith("model.diffusion_model.")
        }
        state_dict = filtered_state_dict

    info = model.load_state_dict(state_dict, strict=False, assign=True)
    logger.info(f"Loaded Lumina: {info}")
    return model

def load_gemma2(
    ckpt_path: Optional[str],
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> Gemma2Model:
    """
    Load the Gemma2 model from the checkpoint path.

    Args:
        ckpt_path (str): Path to the checkpoint.
        dtype (torch.dtype): The data type for the model.
        device (Union[str, torch.device]): The device to load the model on.
        disable_mmap (bool, optional): Whether to disable mmap. Defaults to False.
        state_dict (Optional[dict], optional): The state dict to load. Defaults to None.

    Returns:
        gemma2 (Gemma2Model): The loaded model
    """
    logger.info("Building Gemma2")
    GEMMA2_CONFIG = {
        "_name_or_path": "google/gemma-2-2b",
        "architectures": ["Gemma2Model"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_logit_softcapping": 50.0,
        "bos_token_id": 2,
        "cache_implementation": "hybrid",
        "eos_token_id": 1,
        "final_logit_softcapping": 30.0,
        "head_dim": 256,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 2304,
        "initializer_range": 0.02,
        "intermediate_size": 9216,
        "max_position_embeddings": 8192,
        "model_type": "gemma2",
        "num_attention_heads": 8,
        "num_hidden_layers": 26,
        "num_key_value_heads": 4,
        "pad_token_id": 0,
        "query_pre_attn_scalar": 256,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "sliding_window": 4096,
        "torch_dtype": "float32",
        "transformers_version": "4.44.2",
        "use_cache": True,
        "vocab_size": 256000,
    }

    config = Gemma2Config(**GEMMA2_CONFIG)
    with init_empty_weights():
        gemma2 = Gemma2Model._from_config(config)

    if state_dict is not None:
        sd = state_dict
    else:
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

    for key in list(sd.keys()):
        new_key = key.replace("model.", "")
        if new_key == key:
            break  # the model doesn't have annoying prefix
        sd[new_key] = sd.pop(key)

    # Neta-Lumina support
    if "text_encoders.gemma2_2b.logit_scale" in sd:
        # remove "text_encoders.gemma2_2b.transformer.model." prefix
        filtered_sd = {
            k.replace("text_encoders.gemma2_2b.transformer.model.", ""): v
            for k, v in sd.items()
            if k.startswith("text_encoders.gemma2_2b.transformer.model.")
        }
        sd = filtered_sd

    info = gemma2.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded Gemma2: {info}")
    return gemma2

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    
@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None

configs = {
    "dev": ModelSpec(
        # repo_id="black-forest-labs/FLUX.1-dev",
        # repo_flow="flux1-dev.sft",
        # repo_ae="ae.sft",
        ckpt_path=None,  # os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=None,  # os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "schnell": ModelSpec(
        # repo_id="black-forest-labs/FLUX.1-schnell",
        # repo_flow="flux1-schnell.sft",
        # repo_ae="ae.sft",
        ckpt_path=None,  # os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=None,  # os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}

def load_ae(
    ckpt_path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
) -> AutoEncoder:
    """
    Load the AutoEncoder model from the checkpoint path.

    Args:
        ckpt_path (str): Path to the checkpoint.
        dtype (torch.dtype): The data type for the model.
        device (Union[str, torch.device]): The device to load the model on.
        disable_mmap (bool, optional): Whether to disable mmap. Defaults to False.

    Returns:
        ae (flux_models.AutoEncoder): The loaded model.
    """
    logger.info("Building AutoEncoder")
    with torch.device("meta"):
        # dev and schnell have the same AE params
        ae = AutoEncoder(configs["schnell"].ae_params).to(dtype)

    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=device, disable_mmap=disable_mmap, dtype=dtype)

    # Neta-Lumina support
    if "vae.decoder.conv_in.bias" in sd:
        # remove "vae." prefix
        filtered_sd = {k.replace("vae.", ""): v for k, v in sd.items() if k.startswith("vae.")}
        sd = filtered_sd

    info = ae.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded AE: {info}")
    return ae

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    logger.info("Loading models...")
    device = get_preferred_device()
    if args.device:
        device = torch.device(args.device)

    # Load Lumina DiT model
    model = load_lumina_model(
        args.pretrained_model_name_or_path,
        dtype=None,  # Load in fp32 and then convert
        device="cpu",
        use_flash_attn=args.use_flash_attn,
        use_sage_attn=args.use_sage_attn,
    )

    # Load Gemma2
    gemma2 = load_gemma2(args.gemma2_path, dtype=None, device="cpu")

    # Load Autoencoder
    ae = load_ae(args.ae_path, dtype=None, device="cpu")

    # LoRA
    lora_models = []
    for weights_file in args.lora_weights:
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = 1.0

        weights_sd = load_file(weights_file)
        lora_model, _ = lora_lumina.create_network_from_weights(multiplier, None, ae, [gemma2], model, weights_sd, True)

        if args.merge_lora_weights:
            lora_model.merge_to([gemma2], model, weights_sd)
        else:
            lora_model.apply_to([gemma2], model)
            info = lora_model.load_state_dict(weights_sd, strict=True)
            logger.info(f"Loaded LoRA weights from {weights_file}: {info}")
            lora_model.to(device)
            lora_model.set_multiplier(multiplier)
            lora_model.eval()

        lora_models.append(lora_model)

    if not args.interactive:
        generate_image(
            model,
            gemma2,
            ae,
            args.prompt,
            args.system_prompt,
            args.seed,
            args.image_width,
            args.image_height,
            args.steps,
            args.guidance_scale,
            args.negative_prompt,
            args,
            args.cfg_trunc_ratio,
            args.renorm_cfg,
        )
    else:
        # Interactive mode loop
        image_width = args.image_width
        image_height = args.image_height
        steps = args.steps
        guidance_scale = args.guidance_scale
        cfg_trunc_ratio = args.cfg_trunc_ratio
        renorm_cfg = args.renorm_cfg

        print("Entering interactive mode.")
        while True:
            print(
                "\nEnter prompt (or 'exit'). Options: --w <int> --h <int> --s <int> --d <int> --g <float> --n <str> --ctr <float> --rcfg <float> --m <m1,m2...>"
            )
            user_input = input()
            if user_input.lower() == "exit":
                break
            if not user_input:
                continue

            # Parse options
            options = user_input.split("--")
            prompt = options[0].strip()

            # Set defaults for each generation
            seed = None  # New random seed each time unless specified
            negative_prompt = args.negative_prompt  # Reset to default

            for opt in options[1:]:
                try:
                    opt = opt.strip()
                    if not opt:
                        continue

                    key, value = (opt.split(None, 1) + [""])[:2]

                    if key == "w":
                        image_width = int(value)
                    elif key == "h":
                        image_height = int(value)
                    elif key == "s":
                        steps = int(value)
                    elif key == "d":
                        seed = int(value)
                    elif key == "g":
                        guidance_scale = float(value)
                    elif key == "n":
                        negative_prompt = value if value != "-" else ""
                    elif key == "ctr":
                        cfg_trunc_ratio = float(value)
                    elif key == "rcfg":
                        renorm_cfg = float(value)
                    elif key == "m":
                        multipliers = value.split(",")
                        if len(multipliers) != len(lora_models):
                            logger.error(f"Invalid number of multipliers, expected {len(lora_models)}")
                            continue
                        for i, lora_model in enumerate(lora_models):
                            lora_model.set_multiplier(float(multipliers[i].strip()))
                    else:
                        logger.warning(f"Unknown option: --{key}")

                except (ValueError, IndexError) as e:
                    logger.error(f"Invalid value for option --{key}: '{value}'. Error: {e}")

            generate_image(
                model,
                gemma2,
                ae,
                prompt,
                args.system_prompt,
                seed,
                image_width,
                image_height,
                steps,
                guidance_scale,
                negative_prompt,
                args,
                cfg_trunc_ratio,
                renorm_cfg,
            )

    logger.info("Done.")
