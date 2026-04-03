"""
FreeFuse Mask From Image

Convert user-provided masks (ComfyUI MASK type) into FREEFUSE_MASKS,
bypassing Phase 1 entirely. Each mask corresponds to one LoRA concept.

Two strategies for strengthening LoRA effect with tight masks:
- floor: base LoRA level everywhere, full strength in mask (keeps geometry)
- dilation: expand mask boundary outward (changes geometry)
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
    Use floor and/or dilation to strengthen LoRA effect with tight masks.
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
        optional["floor"] = (
            "FLOAT",
            {
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "tooltip": (
                    "Minimum LoRA strength outside the mask region. "
                    "0.0 = hard mask (LoRA off outside). "
                    "0.3 = 30% LoRA everywhere, 100% inside mask. "
                    "Prevents attention dilution without changing mask shape."
                ),
            },
        )
        optional["dilation"] = (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 100,
                "step": 1,
                "tooltip": (
                    "Expand mask boundary by N pixels (latent space). "
                    "0 = no expansion. "
                    "10-20 = moderate expansion. "
                    "Changes mask shape — use floor instead to keep geometry."
                ),
            },
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
        "Use floor to set base LoRA level everywhere (keeps mask shape). "
        "Use dilation to expand mask boundaries (changes shape)."
    )

    def convert(self, freefuse_data, mask_1, latent=None, floor=0.0,
                dilation=0, **kwargs):
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

        # Build concept masks at latent resolution
        concept_names = []
        concept_masks = []
        for i, adapter_info in enumerate(adapters):
            name = adapter_info.get("name")
            if not name:
                continue
            if i < len(input_masks):
                resized = self._resize_mask(input_masks[i], target_h, target_w)
                concept_names.append(name)
                concept_masks.append(resized)
            else:
                print(
                    f"[FreeFuse MaskFromImage] No mask for adapter '{name}', skipping"
                )

        if not concept_masks:
            return ({"masks": {}},)

        # Apply dilation if requested
        if dilation > 0:
            concept_masks = self._dilate_masks(concept_masks, dilation)

        # Resolve overlaps: where masks overlap after dilation,
        # assign to the concept whose original seed was closer
        if dilation > 0 and len(concept_masks) > 1:
            concept_masks = self._resolve_overlaps(
                concept_masks,
                [self._resize_mask(input_masks[i], target_h, target_w)
                 for i in range(len(concept_masks))],
                target_h, target_w,
            )

        # Apply floor: raise minimum mask value outside masked regions
        if floor > 0.0:
            concept_masks = self._apply_floor(concept_masks, floor)

        # Build output
        mask_dict = {}
        covered = torch.zeros(target_h, target_w)
        for name, mask in zip(concept_names, concept_masks):
            mask_dict[name] = mask
            covered = torch.max(covered, mask)
            coverage_full = (mask >= 1.0).sum() / (target_h * target_w) * 100
            mean_val = mask.mean() * 100
            print(
                f"[FreeFuse MaskFromImage] mask → '{name}' "
                f"({target_h}x{target_w}, {coverage_full:.1f}% full, "
                f"{mean_val:.1f}% mean strength)"
            )

        # Background = regions not covered by any concept at full strength
        # With floor > 0, background gets reduced since concepts have base presence
        bg_mask = torch.ones(target_h, target_w)
        for mask in concept_masks:
            bg_mask = bg_mask * (1.0 - mask)
        bg_mask = bg_mask.clamp(0, 1)
        mask_dict["_background_"] = bg_mask
        bg_mean = bg_mask.mean() * 100
        print(f"[FreeFuse MaskFromImage] '_background_': {bg_mean:.1f}% mean strength")

        return ({"masks": mask_dict},)

    def _apply_floor(self, masks, floor):
        """Raise minimum mask value to floor. Inside mask stays 1.0,
        outside becomes floor instead of 0.0."""
        result = []
        for mask in masks:
            floored = torch.where(mask > 0.5, torch.ones_like(mask),
                                  torch.full_like(mask, floor))
            result.append(floored)
        return result

    def _dilate_masks(self, masks, dilation):
        """Expand each mask by `dilation` pixels using max pooling."""
        kernel = 2 * dilation + 1
        dilated = []
        for mask in masks:
            m = mask.unsqueeze(0).unsqueeze(0)
            m = F.max_pool2d(m, kernel_size=kernel, stride=1,
                             padding=dilation)
            dilated.append(m.squeeze(0).squeeze(0))
        return dilated

    def _resolve_overlaps(self, dilated_masks, seed_masks, h, w):
        """Where dilated masks overlap, assign pixel to nearest seed concept."""
        stacked = torch.stack(dilated_masks, dim=0)
        overlap = (stacked.sum(dim=0) > 1).float()

        if overlap.sum() == 0:
            return dilated_masks

        seeds = torch.stack(seed_masks, dim=0)
        C = seeds.shape[0]
        max_dist = float(h + w)
        distances = torch.where(
            seeds > 0.5,
            torch.zeros(C, h, w),
            torch.full((C, h, w), max_dist),
        )
        dist = distances.unsqueeze(0)
        for _ in range(h + w):
            pooled = -F.max_pool2d(-dist, kernel_size=3, stride=1, padding=1)
            dist = torch.min(dist, pooled + 1.0)
            if (dist < max_dist).all():
                break
        distances = dist.squeeze(0)

        nearest = distances.argmin(dim=0)
        result = []
        for i, mask in enumerate(dilated_masks):
            resolved = mask.clone()
            overlap_region = overlap > 0
            resolved[overlap_region] = (nearest[overlap_region] == i).float()
            result.append(resolved)

        return result

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
