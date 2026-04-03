"""
FreeFuse Attention Bias Only Applicator

Applies attention bias from masks WITHOUT spatial LoRA masking.
LoRA runs at full strength through all layers; spatial control comes
entirely from attention bias guiding which text tokens attend to which
image regions.

Use this instead of FreeFuseMaskApplicator when your masks are tight
(small coverage) and you want to avoid LoRA dilution from hard masking.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, Optional, List, Tuple

import comfy.model_patcher

from ..freefuse_core.attention_bias import AttentionBiasConfig
from ..freefuse_core.attention_bias_patch import apply_attention_bias_patches
from ..freefuse_core.tensor_debug import tensor_scalar_to_float
from ..freefuse_core.token_utils import detect_model_type


class FreeFuseAttentionBiasApplicator:
    """
    Apply attention bias from masks without hard spatial LoRA masking.

    LoRA stays at full strength everywhere. Spatial control comes from
    attention bias only — guiding cross-attention between text tokens
    and image regions based on your masks.

    Avoids the attention dilution problem that occurs when hard-masking
    LoRA output with tight/small masks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "masks": ("FREEFUSE_MASKS",),
                "freefuse_data": ("FREEFUSE_DATA",),
            },
            "optional": {
                "latent": ("LATENT", {
                    "tooltip": "Optional latent for size reference"
                }),
                "bias_scale": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Strength of negative bias (suppress cross-concept attention)"
                }),
                "positive_bias_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Strength of positive bias (enhance same-concept attention)"
                }),
                "bidirectional": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply bias for both image→text and text→image directions"
                }),
                "use_positive_bias": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add positive bias for same-concept pairs"
                }),
                "bias_blocks": (
                    ["all", "double_stream_only", "single_stream_only",
                     "last_half_double", "last_half", "none"],
                    {
                        "default": "double_stream_only",
                        "tooltip": "Which transformer blocks to apply attention bias"
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_bias"
    CATEGORY = "FreeFuse"

    DESCRIPTION = (
        "Apply attention bias from masks WITHOUT hard spatial LoRA masking. "
        "LoRA runs at full strength through all layers. "
        "Spatial control comes from attention bias guiding cross-attention. "
        "Use this for tight masks to avoid LoRA dilution."
    )

    def apply_bias(
        self,
        model,
        masks,
        freefuse_data,
        latent=None,
        bias_scale=5.0,
        positive_bias_scale=1.0,
        bidirectional=True,
        use_positive_bias=True,
        bias_blocks="double_stream_only",
    ):
        mask_dict = masks.get("masks", {})
        adapters = freefuse_data.get("adapters", [])
        token_pos_maps = freefuse_data.get("token_pos_maps", {})

        if not mask_dict:
            print("[FreeFuse BiasOnly] Warning: No masks provided")
            return (model,)

        if not adapters:
            print("[FreeFuse BiasOnly] Warning: No adapters registered")
            return (model,)

        # Determine latent size
        latent_size = self._get_latent_size(mask_dict, latent)
        if latent_size is None:
            print("[FreeFuse BiasOnly] Warning: Could not determine latent size")
            return (model,)

        model_clone = model.clone()

        model_type = self._detect_model_type(
            model_clone,
            model_type_hint=freefuse_data.get("model_type"),
        )
        print(f"[FreeFuse BiasOnly] Detected model type: {model_type}")
        print(f"[FreeFuse BiasOnly] Latent size: {latent_size}")
        print(f"[FreeFuse BiasOnly] Mode: attention bias only (no spatial LoRA masking)")

        # Log mask coverage
        for name, mask in mask_dict.items():
            if name.startswith("_"):
                continue
            coverage = tensor_scalar_to_float(mask.sum()) / mask.numel() * 100
            print(f"[FreeFuse BiasOnly]   {name}: {coverage:.1f}% coverage")

        # Apply attention bias only (no _apply_masks_to_model)
        if bias_blocks != "none":
            self._apply_attention_bias(
                model_clone,
                mask_dict,
                token_pos_maps,
                latent_size,
                model_type,
                bias_scale=bias_scale,
                positive_bias_scale=positive_bias_scale,
                bidirectional=bidirectional,
                use_positive_bias=use_positive_bias,
                bias_blocks=bias_blocks,
            )

        return (model_clone,)

    def _detect_model_type(self, model_patcher, model_type_hint=None):
        model_type = detect_model_type(
            model=model_patcher, model_type_hint=model_type_hint,
        )
        return "sdxl" if model_type == "unknown" else model_type

    def _get_latent_size(self, mask_dict, latent):
        # Priority 1: mask dimensions
        for mask in mask_dict.values():
            if mask.dim() == 2:
                return (mask.shape[0], mask.shape[1])
            elif mask.dim() == 3:
                return (mask.shape[1], mask.shape[2])

        # Priority 2: latent input
        if latent is not None:
            samples = latent.get("samples")
            if samples is not None:
                return (samples.shape[2], samples.shape[3])

        return None

    def _apply_attention_bias(
        self,
        model_patcher,
        mask_dict,
        token_pos_maps,
        latent_size,
        model_type,
        bias_scale,
        positive_bias_scale,
        bidirectional,
        use_positive_bias,
        bias_blocks,
    ):
        config = AttentionBiasConfig(
            enabled=True,
            bias_scale=bias_scale,
            positive_bias_scale=positive_bias_scale,
            bidirectional=bidirectional,
            use_positive_bias=use_positive_bias,
            apply_to_blocks=bias_blocks if bias_blocks != "all" else None,
        )

        latent_h, latent_w = latent_size

        if model_type in ("flux", "flux2"):
            img_seq_len = latent_h * latent_w

            txt_seq_len = 512
            for positions_list in token_pos_maps.values():
                if positions_list and positions_list[0]:
                    max_pos = max(positions_list[0])
                    txt_seq_len = max(txt_seq_len, max_pos + 10)

            lora_masks_flat = {}
            for name, mask in mask_dict.items():
                if name.startswith("_"):
                    continue
                if mask.dim() == 3:
                    mask = mask[0]
                mask_flat = mask.reshape(-1)
                lora_masks_flat[name] = mask_flat.unsqueeze(0)

            apply_attention_bias_patches(
                model_patcher=model_patcher,
                attention_bias=None,
                config=config,
                txt_seq_len=txt_seq_len,
                model_type="flux",
                lora_masks=lora_masks_flat,
                token_pos_maps=token_pos_maps,
            )
            print(
                f"[FreeFuse BiasOnly] Applied attention bias for {model_type} "
                f"(bias_scale={bias_scale}, positive_scale={positive_bias_scale}, "
                f"bidirectional={bidirectional}, blocks={bias_blocks})"
            )

        elif model_type == "z_image":
            img_seq_len = latent_h * latent_w

            cap_seq_len = 256
            for positions_list in token_pos_maps.values():
                if positions_list and positions_list[0]:
                    max_pos = max(positions_list[0])
                    cap_seq_len = max(cap_seq_len, max_pos + 10)

            lora_masks_flat = {}
            for name, mask in mask_dict.items():
                if name.startswith("_"):
                    continue
                if mask.dim() == 3:
                    mask = mask[0]
                mask_flat = mask.reshape(-1)
                lora_masks_flat[name] = mask_flat.unsqueeze(0)

            apply_attention_bias_patches(
                model_patcher=model_patcher,
                attention_bias=None,
                config=config,
                txt_seq_len=cap_seq_len,
                model_type="z_image",
                lora_masks=lora_masks_flat,
                token_pos_maps=token_pos_maps,
            )
            print(
                f"[FreeFuse BiasOnly] Applied attention bias for Z-Image "
                f"(bias_scale={bias_scale}, positive_scale={positive_bias_scale})"
            )

        else:  # SDXL
            apply_attention_bias_patches(
                model_patcher=model_patcher,
                attention_bias=None,
                config=config,
                txt_seq_len=77,
                model_type="sdxl",
                lora_masks={
                    name: mask.reshape(-1).unsqueeze(0)
                    for name, mask in mask_dict.items()
                    if not name.startswith("_")
                },
                token_pos_maps=token_pos_maps,
            )
            print(f"[FreeFuse BiasOnly] Applied attention bias for SDXL")


NODE_CLASS_MAPPINGS = {
    "FreeFuseAttentionBiasApplicator": FreeFuseAttentionBiasApplicator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseAttentionBiasApplicator": "FreeFuse Attention Bias Only",
}
