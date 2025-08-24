import glob
import os
from typing import Any, List, Optional, Tuple, Union
import gc

import torch
from transformers import AutoTokenizer, AutoModel, Gemma2Model, GemmaTokenizerFast
from strategy_base import (
    LatentsCachingStrategy,
    TokenizeStrategy,
    TextEncodingStrategy,
    TextEncoderOutputsCachingStrategy,
)
from PIL import Image
import numpy as np
import sys
import logging

def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()

class ImageInfo:
    def __init__(self, image_key: str, num_repeats: int, caption: str, is_reg: bool, absolute_path: str) -> None:
        self.image_key: str = image_key
        self.num_repeats: int = num_repeats
        self.caption: str = caption
        self.is_reg: bool = is_reg
        self.absolute_path: str = absolute_path
        self.image_size: Tuple[int, int] = None
        self.resized_size: Tuple[int, int] = None
        self.bucket_reso: Tuple[int, int] = None
        self.latents: Optional[torch.Tensor] = None
        self.latents_flipped: Optional[torch.Tensor] = None
        self.latents_npz: Optional[str] = None  # set in cache_latents
        self.latents_original_size: Optional[Tuple[int, int]] = None  # original image size, not latents size
        self.latents_crop_ltrb: Optional[Tuple[int, int]] = (
            None  # crop left top right bottom in original pixel size, not latents size
        )
        self.cond_img_path: Optional[str] = None
        self.image: Optional[Image.Image] = None  # optional, original PIL Image
        self.text_encoder_outputs_npz: Optional[str] = None  # set in cache_text_encoder_outputs

        # new
        self.text_encoder_outputs: Optional[List[torch.Tensor]] = None
        # old
        self.text_encoder_outputs1: Optional[torch.Tensor] = None
        self.text_encoder_outputs2: Optional[torch.Tensor] = None
        self.text_encoder_pool2: Optional[torch.Tensor] = None

        self.alpha_mask: Optional[torch.Tensor] = None  # alpha mask can be flipped in runtime
        self.resize_interpolation: Optional[str] = None

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

setup_logging()
import logging

logger = logging.getLogger(__name__)


GEMMA_ID = "google/gemma-2-2b"


class LuminaTokenizeStrategy(TokenizeStrategy):
    def __init__(
        self, system_prompt:str, max_length: Optional[int], tokenizer_cache_dir: Optional[str] = None
    ) -> None:
        self.tokenizer: GemmaTokenizerFast = AutoTokenizer.from_pretrained(
            GEMMA_ID, cache_dir=tokenizer_cache_dir
        )
        self.tokenizer.padding_side = "right"

        if system_prompt is None:
            system_prompt = ""
        system_prompt_special_token = "<Prompt Start>"
        system_prompt = f"{system_prompt} {system_prompt_special_token} " if system_prompt else ""
        self.system_prompt = system_prompt

        if max_length is None:
            self.max_length = 256
        else:
            self.max_length = max_length

    def tokenize(
        self, text: Union[str, List[str]], is_negative: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text (Union[str, List[str]]): Text to tokenize

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                token input ids, attention_masks
        """
        text = [text] if isinstance(text, str) else text
        
        # In training, we always add system prompt (is_negative=False)
        if not is_negative:
            # Add system prompt to the beginning of each text
            text = [self.system_prompt + t for t in text]

        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            pad_to_multiple_of=8,
        )
        return (encodings.input_ids, encodings.attention_mask)

    def tokenize_with_weights(
        self, text: str | List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            text (Union[str, List[str]]): Text to tokenize

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
                token input ids, attention_masks, weights
        """
        # Gemma doesn't support weighted prompts, return uniform weights
        tokens, attention_masks = self.tokenize(text)
        weights = [torch.ones_like(t) for t in tokens]
        return tokens, attention_masks, weights


class LuminaTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self) -> None:
        super().__init__()

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tokenize_strategy (LuminaTokenizeStrategy): Tokenize strategy
            models (List[Any]): Text encoders
            tokens (Tuple[torch.Tensor, torch.Tensor]): tokens, attention_masks

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                hidden_states, input_ids, attention_masks
        """
        text_encoder = models[0]
        # Check model or torch dynamo OptimizedModule
        assert isinstance(text_encoder, Gemma2Model) or isinstance(text_encoder._orig_mod, Gemma2Model), f"text encoder is not Gemma2Model {text_encoder.__class__.__name__}"
        input_ids, attention_masks = tokens

        outputs = text_encoder(
            input_ids=input_ids.to(text_encoder.device),
            attention_mask=attention_masks.to(text_encoder.device),
            output_hidden_states=True,
            return_dict=True,
        )

        return outputs.hidden_states[-2], input_ids, attention_masks

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: Tuple[torch.Tensor, torch.Tensor],
        weights: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tokenize_strategy (LuminaTokenizeStrategy): Tokenize strategy
            models (List[Any]): Text encoders
            tokens (Tuple[torch.Tensor, torch.Tensor]): tokens, attention_masks
            weights_list (List[torch.Tensor]): Currently unused

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                hidden_states, input_ids, attention_masks
        """
        # For simplicity, use uniform weighting
        return self.encode_tokens(tokenize_strategy, models, tokens)


class LuminaTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    LUMINA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_lumina_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
    ) -> None:
        super().__init__(
            cache_to_disk,
            batch_size,
            skip_disk_cache_validity_check,
            is_partial,
        )

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return (
            os.path.splitext(image_abs_path)[0]
            + LuminaTextEncoderOutputsCachingStrategy.LUMINA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX
        )

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        """
        Args:
            npz_path (str): Path to the npz file.

        Returns:
            bool: True if the npz file is expected to be cached.
        """
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "hidden_state" not in npz:
                return False
            if "attention_mask" not in npz:
                return False
            if "input_ids" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        """
        Load outputs from a npz file

        Returns:
            List[np.ndarray]: hidden_state, input_ids, attention_mask
        """
        data = np.load(npz_path)
        hidden_state = data["hidden_state"]
        attention_mask = data["attention_mask"]
        input_ids = data["input_ids"]
        return [hidden_state, input_ids, attention_mask]

    @torch.no_grad()
    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        batch: List[ImageInfo],
    ) -> None:
        """
        Args:
            tokenize_strategy (LuminaTokenizeStrategy): Tokenize strategy
            models (List[Any]): Text encoders
            text_encoding_strategy (LuminaTextEncodingStrategy):
            infos (List): List of ImageInfo

        Returns:
            None
        """
        assert isinstance(text_encoding_strategy, LuminaTextEncodingStrategy)
        assert isinstance(tokenize_strategy, LuminaTokenizeStrategy)

        captions = [info.caption for info in batch]

        if self.is_weighted:
            tokens, attention_masks, weights_list = (
                tokenize_strategy.tokenize_with_weights(captions)
            )
            hidden_state, input_ids, attention_masks = (
                text_encoding_strategy.encode_tokens_with_weights(
                    tokenize_strategy,
                    models,
                    (tokens, attention_masks),
                    weights_list,
                )
            )
        else:
            tokens = tokenize_strategy.tokenize(captions)
            hidden_state, input_ids, attention_masks = (
                text_encoding_strategy.encode_tokens(
                    tokenize_strategy, models, tokens
                )
            )

        if hidden_state.dtype != torch.float32:
            hidden_state = hidden_state.float()

        hidden_state = hidden_state.cpu().numpy()
        attention_mask = attention_masks.cpu().numpy() # (B, S)
        input_ids = input_ids.cpu().numpy() # (B, S) 


        for i, info in enumerate(batch):
            hidden_state_i = hidden_state[i]
            attention_mask_i = attention_mask[i]
            input_ids_i = input_ids[i]

            if self.cache_to_disk:
                assert info.text_encoder_outputs_npz is not None, f"Text encoder cache outputs to disk not found for image {info.image_key}"
                np.savez(
                    info.text_encoder_outputs_npz,
                    hidden_state=hidden_state_i,
                    attention_mask=attention_mask_i,
                    input_ids=input_ids_i,
                )
            else:
                info.text_encoder_outputs = [
                    hidden_state_i,
                    input_ids_i,
                    attention_mask_i,
                ]


class LuminaLatentsCachingStrategy(LatentsCachingStrategy):
    LUMINA_LATENTS_NPZ_SUFFIX = "_lumina.npz"

    def __init__(
        self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return LuminaLatentsCachingStrategy.LUMINA_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(
        self, absolute_path: str, image_size: Tuple[int, int]
    ) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + LuminaLatentsCachingStrategy.LUMINA_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(
        self,
        bucket_reso: Tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        alpha_mask: bool,
    ) -> bool:
        """
        Args:
            bucket_reso (Tuple[int, int]): The resolution of the bucket.
            npz_path (str): Path to the npz file.
            flip_aug (bool): Whether to flip the image.
            alpha_mask (bool): Whether to apply
        """
        return self._default_is_disk_cached_latents_expected(
            8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True
        )

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[List[int]],
        Optional[List[int]],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Args:
            npz_path (str): Path to the npz file.
            bucket_reso (Tuple[int, int]): The resolution of the bucket.

        Returns:
            Tuple[
                Optional[np.ndarray],
                Optional[List[int]],
                Optional[List[int]],
                Optional[np.ndarray],
                Optional[np.ndarray],
            ]: Tuple of latent tensors, attention_mask, input_ids, latents, latents_unet
        """
        return self._default_load_latents_from_disk(
            8, npz_path, bucket_reso
        )  # support multi-resolution

    # TODO remove circular dependency for ImageInfo
    def cache_batch_latents(
        self,
        model,
        batch: List,
        flip_aug: bool,
        alpha_mask: bool,
        random_crop: bool,
    ):
        encode_by_vae = lambda img_tensor: model.encode(img_tensor).to("cpu")
        vae_device = model.device
        vae_dtype = model.dtype

        self._default_cache_batch_latents(
            encode_by_vae,
            vae_device,
            vae_dtype,
            batch,
            flip_aug,
            alpha_mask,
            random_crop,
            multi_resolution=True,
        )

        clean_memory_on_device(model.device)
