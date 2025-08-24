import torch
import logging
from typing import Callable, Dict, Any
try:
    from k_diffusion import sampling as k_sampling
except ImportError:
    raise ImportError("k-diffusion库未安装。请运行 'pip install k-diffusion' 进行安装。")

import schedulers as k_schedulers

import lumina_models

logger = logging.getLogger(__name__)

def time_snr_shift(shift, t):
    """一个辅助函数，执行逻辑中核心的非线性时间转换"""
    t = torch.clamp(t, 0.0, 1.0)
    if shift == 1.0:
        return t
    return shift * t / (1 + (shift - 1) * t)

class ModelSamplingDiscreteFlow(torch.nn.Module):
    """
    Sigma 到 Timestep 的转换逻辑
    注意：此类仅用于生成 sigma 列表，其内部的 timestep() 方法在适配器中不会被使用
    """
    def __init__(self, shift=1.0, timesteps=1000, multiplier=1000.0):
        super().__init__()
        self.set_parameters(shift=shift, timesteps=timesteps, multiplier=multiplier)

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000.0):
        self.shift = shift
        self.multiplier = float(multiplier)
        
        # sigmas 列表
        t_values = torch.arange(1, timesteps + 1, dtype=torch.float32)
        ts = self.sigma(t_values)
        # k-diffusion调度器期望sigma是降序的
        self.register_buffer('sigmas', ts.flip(0))

    @property
    def sigma_min(self):
        return self.sigmas[-1]

    @property
    def sigma_max(self):
        return self.sigmas[0]

    def timestep(self, sigma):
        # 实现正确的 sigma -> timestep 逆运算
        if self.shift == 1.0:
            t_normalized = sigma
        else:
            # 这是 time_snr_shift 的逆运算
            t_normalized = sigma / (self.shift - sigma * (self.shift - 1))
        return t_normalized * self.multiplier

    def sigma(self, timestep):
        normalized_timestep = timestep / self.multiplier
        return time_snr_shift(self.shift, normalized_timestep)



# 模型包装器
class KDiffusionModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model: lumina_models.NextDiT,
        model_sampling: torch.nn.Module,
        guidance_scale: float,
        cfg_trunc_ratio: float,
        renorm_cfg: float,
        text_encoder_hidden_states: torch.Tensor,
        text_encoder_attention_mask: torch.Tensor,
        neg_hidden_states: torch.Tensor,
        neg_attention_mask: torch.Tensor,
    ):
        super().__init__()
        self.inner_model = model
        # 我们需要从 model_sampling 对象中获取 shift 值
        self.shift = model_sampling.shift
        self.guidance_scale = guidance_scale
        self.cfg_trunc_ratio = cfg_trunc_ratio
        self.renorm_cfg = renorm_cfg
        self.cond = {
            "cap_feats": text_encoder_hidden_states,
            "cap_mask": text_encoder_attention_mask.to(dtype=torch.int32),
        }
        self.uncond = {
            "cap_feats": neg_hidden_states,
            "cap_mask": neg_attention_mask.to(dtype=torch.int32),
        }

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs):
        batch_size = x.shape[0]
        if sigma.shape[0] != batch_size:
            sigma = sigma.repeat(batch_size)

        #print(f"[DEBUG] Input x stats: mean={x.mean().item():.6f}, std={x.std().item():.6f}, shape={x.shape}")
        #print(f"[DEBUG] sigma: {sigma.item():.6f}")

        # k_diffusion的sigma实际上就是经过shift变换后的normalized timestep
        # 所以直接将sigma当作t_normalized，然后计算current_timestep
        
        # 在原始scheduler中：current_timestep = 1 - t / num_train_timesteps
        # 而k_diffusion的sigma就是经过shift后的t / num_train_timesteps
        current_timestep = 1.0 - sigma
        current_timestep = current_timestep.to(device=x.device, dtype=x.dtype)

        #print(f"[DEBUG] current_timestep: {current_timestep.item():.6f}")

        self.inner_model.prepare_block_swap_before_forward()
        
        v_pred_cond = self.inner_model(
            x, current_timestep, **self.cond
        )

        #print(f"[DEBUG] v_pred_cond stats: mean={v_pred_cond.mean().item():.6f}, std={v_pred_cond.std().item():.6f}")

        if current_timestep[0] < self.cfg_trunc_ratio:
            self.inner_model.prepare_block_swap_before_forward()
            v_pred_uncond = self.inner_model(
                x, current_timestep, **self.uncond
            )
            v_pred = v_pred_uncond + self.guidance_scale * (
                v_pred_cond - v_pred_uncond
            )
            if float(self.renorm_cfg) > 0.0:
                cond_norm = torch.linalg.vector_norm(v_pred_cond, dim=tuple(range(1, len(v_pred_cond.shape))), keepdim=True)
                max_new_norms = cond_norm * float(self.renorm_cfg)
                noise_norms = torch.linalg.vector_norm(v_pred, dim=tuple(range(1, len(v_pred.shape))), keepdim=True)
                for i, (noise_norm, max_new_norm) in enumerate(zip(noise_norms, max_new_norms)):
                    if noise_norm > max_new_norm:
                        v_pred[i] = v_pred[i] * (max_new_norm / noise_norm)
        else:
            v_pred = v_pred_cond
        
        # 根据原始main.py中的negative sign
        # noise_pred = -noise_pred, 然后 denoised = sample - model_output * sigma
        # 等效于：denoised = sample - (-v_pred) * sigma = sample + v_pred * sigma
        denoised = x + v_pred * sigma.view(-1, 1, 1, 1)

        #print(f"[DEBUG] Final denoised stats: mean={denoised.mean().item():.6f}, std={denoised.std().item():.6f}")

        return denoised

# 主适配器函数
def sample_with_k_diffusion(
    model: lumina_models.NextDiT,
    latents: torch.Tensor,
    steps: int,
    guidance_scale: float,
    cfg_trunc_ratio: float,
    renorm_cfg: float,
    sampler: str,
    scheduler_func: str,
    shift: float,
    text_encoder_hidden_states: torch.Tensor,
    text_encoder_attention_mask: torch.Tensor,
    neg_hidden_states: torch.Tensor,
    neg_attention_mask: torch.Tensor,
    **sampler_kwargs,
):
    device = latents.device
    logger.info(f"Using k-diffusion sampler: '{sampler}' with scheduler: '{scheduler_func}'")
    
    # 初始化 sigma/timestep 转换器
    model_sampling = ModelSamplingDiscreteFlow(shift=shift)
    model_sampling.to(device)
    logger.info(f"Initialized model sampling with shift={shift}")

    scheduler_fn = getattr(k_schedulers, scheduler_func, None)
    if not scheduler_fn:
        raise ValueError(f"Scheduler function '{scheduler_func}' not found in schedulers.py.")
        
    sigmas = scheduler_fn(model_sampling, steps)
    sigmas = sigmas.to(device)
    logger.info(f"Generated {len(sigmas)} sigmas. Range: {sigmas[0].item():.4f} to {sigmas[-2].item():.4f}")
    #print(f"[DEBUG] Full sigma sequence: {sigmas.cpu().numpy()}")

    wrapped_model = KDiffusionModelWrapper(
        model, model_sampling, guidance_scale, cfg_trunc_ratio, renorm_cfg,
        text_encoder_hidden_states, text_encoder_attention_mask,
        neg_hidden_states, neg_attention_mask,
    )
    
    sampler_fn_name = f"sample_{sampler}"
    sampler_fn = getattr(k_sampling, sampler_fn_name, None)
    if not sampler_fn:
        raise ValueError(f"Sampler '{sampler}' not found in 'k_diffusion.sampling'.")

    # 不要缩放初始噪声
    # 原始scheduler中，输入的latents就是标准的随机噪声，没有额外缩放
    x_T = latents
    #print(f"[DEBUG] Initial latents stats: mean={x_T.mean().item():.6f}, std={x_T.std().item():.6f}, shape={x_T.shape}")

    logger.info("Starting k-diffusion sampling loop...")
    # 强制使用 float32 以保证数值稳定性
    output_latents = sampler_fn(wrapped_model, x_T.to(torch.float32), sigmas.to(torch.float32), disable=False, **sampler_kwargs)
    logger.info("k-diffusion sampling finished.")
    
    #print(f"[DEBUG] Final output stats: mean={output_latents.mean().item():.6f}, std={output_latents.std().item():.6f}")

    return output_latents.to(latents.dtype) # 将输出转换回原始数据类型