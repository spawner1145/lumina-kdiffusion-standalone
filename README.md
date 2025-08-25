# Lumina-KDiffusion-Standalone

本项目为独立版的Lumina模型推理脚本，支持基于DiT架构的Lumina模型、Gemma2文本编码器、以及自动编码器(AutoEncoder)的文本到图像生成。支持普通调度器与[k-diffusion](https://github.com/crowsonkb/k-diffusion)采样方式，兼容LoRA权重、半精度/混合精度推理、多种设备（CUDA、MPS、XPU/Intel GPU）、交互式多轮生成等。

## 特性 Features

- **支持主流推理硬件**：自动检测CUDA、MPS(Apple)、XPU(Intel)等设备，智能选择适配。
- **多种采样方式**：内置原生FlowMatchEulerDiscreteScheduler与k-diffusion采样。
- **LoRA插件支持**：支持加载多个LoRA权重，动态调整融合倍率。
- **模型和编码器高效加载**：支持`safetensors`格式，内置内存高效加载器。
- **交互式生成**：可进入交互模式批量生成图片，动态调整参数。
- **自定义数据类型**：支持bf16、fp16、fp32等多种精度。
- **灵活的prompt结构**：支持正向与负向prompt；可单独设定system prompt。
- **日志美化**：支持Rich日志输出。

## 环境依赖

- Python 3.9+
- PyTorch 2.1+ (支持float8需2.3+)
- [einops](https://github.com/arogozhnikov/einops)
- tqdm
- accelerate
- safetensors
- Pillow
- transformers
- rich (可选, 用于美化日志)
- intel_extension_for_pytorch (可选, Intel GPU)

**安装依赖：**
```bash
pip install torch einops tqdm accelerate safetensors pillow transformers rich
# Intel GPU支持（可选）
pip install intel-extension-for-pytorch
```

## 模型准备

你需要准备以下权重文件：

### 1. Lumina DiT模型
> 推荐使用safetensors格式

- `neta-lumina-v1.0.safetensors`  
  可在[Neta-Lumina项目](https://huggingface.co/neta-art/Neta-Lumina/tree/main/Unet)获取

### 2. Gemma2文本编码器
- `gemma_2_2b_fp16.safetensors`  
  可在[Neta-Lumina项目](https://huggingface.co/neta-art/Neta-Lumina/tree/main/Text%20Encoder)或ComfyUI模型库处获取

### 3. 自动编码器（VAE）
- `ae.safetensors`  
  可在[Neta-Lumina项目](https://huggingface.co/neta-art/Neta-Lumina/tree/main/VAE)或ComfyUI模型库处获取

### 4. (可选) LoRA权重
- LoRA权重需为safetensors格式，格式为`路径;倍率`，如`lora1.safetensors;0.7`。

## 使用方法

### 快速生成

```bash
python main.py \
  --pretrained_model_name_or_path "/path/to/neta-lumina-v1.0.safetensors" \
  --gemma2_path "/path/to/gemma_2_2b_fp16.safetensors" \
  --ae_path "/path/to/ae.safetensors" \
  --prompt "一只在太空行走的柴犬" \
  --output_dir "./outputs" \
  --steps 36 \
  --guidance_scale 3.5 \
  --image_width 1024 \
  --image_height 1024 \
  --dtype bf16
```

**常用参数说明：**

- `--prompt`: 生成图片的正向描述
- `--negative_prompt`: 反向prompt
- `--output_dir`: 输出图片目录
- `--seed`: 随机种子，留空则随机
- `--steps`: 采样步数
- `--guidance_scale`: CFG引导强度
- `--image_width`/`--image_height`: 输出图片尺寸
- `--dtype`: 主模型精度 (bf16 / fp16 / float)
- `--gemma2_dtype`/`--ae_dtype`: 文本编码器和AE的精度
- `--device`: 指定设备，如"cuda:0"、"cpu"
- `--offload`: 启用模型推理时自动卸载节省显存
- `--use_k_diffusion`: 使用k-diffusion采样
- `--lora_weights`: LoRA权重，多个用空格分隔，格式为"path;multiplier"
- `--merge_lora_weights`: 将LoRA直接合并入主模型（不可逆）

### 交互模式

```bash
python main.py [参数同上] --interactive
```
进入交互模式后，可多轮输入prompt，支持如下快捷参数：

- `--w <int>`      图片宽度
- `--h <int>`      图片高度
- `--s <int>`      步数
- `--d <int>`      随机种子
- `--g <float>`    guidance scale
- `--n <str>`      负向prompt（-表示清空）
- `--ctr <float>`  cfg_trunc_ratio
- `--rcfg <float>` renorm_cfg
- `--m <倍率1,倍率2,...>`  动态调整LoRA倍率

**例子：**
```
一只在雪地里奔跑的狼 --w 768 --h 512 --s 28 --g 4.5 --d 123
```

## k-diffusion采样

- 启用方式: `--use_k_diffusion`
- 采样器选择: `--sampler euler|heun|dpm_2|dpm_2_ancestral|euler_ancestral`
- 调度函数: `--scheduler_func normal_scheduler|simple_scheduler|beta_scheduler|linear_quadratic_schedule`
- 参数：`--discrete_flow_shift` 控制FlowMatchEuler shift

## LoRA权重加载

- 支持多个LoRA权重
- 权重倍率可动态调整
- 合并模式下权重会直接写入主模型（适合单独推理/节约显存）

**例子：**
```bash
python main.py ... --lora_weights "lora1.safetensors;0.7" "lora2.safetensors;0.3"
```

## 进阶用法

- 支持float8 safetensors权重加载（需pytorch 2.3+）
- 支持流式大权重内存高效加载
- 支持自定义AutoEncoder参数/模型结构（见源码`configs`）

## 主要依赖与代码结构

- `main.py`: 主推理流程与入口，参数解析，模型加载与推理核心
- `lumina_models.py`: Lumina主模型结构
- `strategy_lumina.py`: Prompt分词、编码器策略
- `ori_schedulers.py`: FlowMatchEulerDiscreteScheduler等调度器
- `lora_lumina.py`: LoRA融合与应用逻辑
- `k_diffusion_adapter.py`: k-diffusion采样适配
- 其余依赖见requirements

## 致谢

- [Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0)
- [Neta-Lumina](https://huggingface.co/neta-art/Neta-Lumina)
- [Gemma2](https://huggingface.co/google/gemma-2-2b)
- [Flux](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [k-diffusion](https://github.com/crowsonkb/k-diffusion)
- [sd-scripts](https://github.com/kohya-ss/sd-scripts.git)
- 以及各开源贡献者

---

## License

本项目仅供研究与学习用途，模型权重请遵循各自授权协议。

如有疑问或需求，欢迎Issue交流。
