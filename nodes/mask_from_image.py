"""
FreeFuse Mask From Image

Convert user-provided masks (ComfyUI MASK type) into FREEFUSE_MASKS,
bypassing Phase 1 entirely. Each mask corresponds to one LoRA concept.

Concept masks define where each LoRA is active. Uncovered regions
become background (no LoRA applied).
"""

import torch
import torch.nn.functional as F


class FreeFuseMaskFromImage:
    """
    Convert per-concept pixel masks into FREEFUSE_MASKS for Phase 2.

    Provide one MASK per LoRA concept. The node reads adapter names from
    freefuse_data and maps them in order: mask_1 → first adapter, etc.
    Masks are resized to latent resolution automatically.

    White = LoRA active, Black = LoRA inactive.
    Uncovered regions become background (no LoRA).
    """

    MAX_CONCEPTS = 6

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "freefuse_data": ("FREEFUSE_DATA",),
            "mask_1": ("MASK",),
        }
        optional = {}
        for i in range(2, cls.MAX_CONCEPTS + 1):
            optional[f"mask_{i}"] = ("MASK",)
        optional["latent"] = (
            "LATENT",
            {"tooltip": "Optional latent for target resolution reference"},
        )
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("FREEFUSE_MASKS",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "convert"
    CATEGORY = "FreeFuse"

    DESCRIPTION = (
        "Convert pixel masks into FreeFuse masks, bypassing Phase 1. "
        "Provide one mask per LoRA concept (in adapter order). "
        "White = LoRA active in that region. "
        "Uncovered regions become background (no LoRA applied)."
    )

    def convert(self, freefuse_data, mask_1, latent=None, **kwargs):
        adapters = freefuse_data.get("adapters", [])
        if not adapters:
            print("[FreeFuse MaskFromImage] Warning: No adapters in freefuse_data")
            return ({"masks": {}},)

        # Collect provided masks in order
        input_masks = [mask_1]
        for i in range(2, self.MAX_CONCEPTS + 1):
            m = kwargs.get(f"mask_{i}")
            if m is not None:
                input_masks.append(m)

        if len(input_masks) < len(adapters):
            print(
                f"[FreeFuse MaskFromImage] Warning: {len(adapters)} adapters but "
                f"only {len(input_masks)} masks provided. Extra adapters will have "
                f"no mask."
            )

        # Determine target latent size
        target_h, target_w = self._get_target_size(input_masks[0], latent)

        # Build mask dict
        mask_dict = {}
        covered = torch.zeros(target_h, target_w)

        for i, adapter_info in enumerate(adapters):
            name = adapter_info.get("name")
            if not name:
                continue
            if i < len(input_masks):
                resized = self._resize_mask(input_masks[i], target_h, target_w)
                mask_dict[name] = resized
                covered = torch.max(covered, resized)
                coverage = resized.sum() / (target_h * target_w) * 100
                print(
                    f"[FreeFuse MaskFromImage] mask_{i + 1} → '{name}' "
                    f"({target_h}x{target_w}, {coverage:.1f}% coverage)"
                )
            else:
                print(
                    f"[FreeFuse MaskFromImage] No mask for adapter '{name}', skipping"
                )

        # Background = uncovered regions
        bg_mask = (1.0 - covered).clamp(0, 1)
        mask_dict["_background_"] = bg_mask
        bg_coverage = bg_mask.sum() / (target_h * target_w) * 100
        print(f"[FreeFuse MaskFromImage] '_background_': {bg_coverage:.1f}% coverage")

        return ({"masks": mask_dict},)

    def _resize_mask(self, mask, target_h, target_w):
        """Resize a ComfyUI MASK tensor to target latent resolution."""
        if mask.dim() == 3:
            mask = mask[0]
        mask = (mask > 0.5).float()
        resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        return (resized > 0.5).float()

    def _get_target_size(self, mask, latent):
        """Determine target latent-space resolution."""
        if latent is not None:
            samples = latent.get("samples")
            if samples is not None:
                return samples.shape[2], samples.shape[3]

        if mask.dim() == 3:
            h, w = mask.shape[1], mask.shape[2]
        else:
            h, w = mask.shape[0], mask.shape[1]

        target_h = max(1, h // 16)
        target_w = max(1, w // 16)
        print(
            f"[FreeFuse MaskFromImage] No latent provided, using 16x downscale: "
            f"{h}x{w} → {target_h}x{target_w}"
        )
        return target_h, target_w


NODE_CLASS_MAPPINGS = {
    "FreeFuseMaskFromImage": FreeFuseMaskFromImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseMaskFromImage": "FreeFuse Mask From Image",
}
