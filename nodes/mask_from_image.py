"""
FreeFuse Mask From Image

Convert user-provided masks (ComfyUI MASK type) into FREEFUSE_MASKS,
bypassing Phase 1 entirely. Each mask corresponds to one LoRA concept.

Matches Phase 1 behavior:
1. Concepts partition the full image among themselves (no background competing)
2. Background is carved out separately from uncovered seed regions
"""

import torch
import torch.nn.functional as F


class FreeFuseMaskFromImage:
    """
    Convert per-concept pixel masks into FREEFUSE_MASKS for Phase 2.

    Provide one MASK per LoRA concept. The node reads adapter names from
    freefuse_data and maps them in order: mask_1 → first adapter, etc.
    Masks are resized to latent resolution automatically.

    By default, masks are expanded to cover the full image (like Phase 1):
    concepts partition among themselves first, then background is carved
    out from originally uncovered regions.
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
        optional["partition_mode"] = (
            ["full_partition", "seed_only"],
            {
                "default": "full_partition",
                "tooltip": (
                    "full_partition: expand masks to cover entire image "
                    "(like Phase 1, recommended). "
                    "seed_only: use masks as-is without expansion"
                ),
            },
        )
        optional["generate_background"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "Include a background mask from originally uncovered regions"
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
        "By default, masks are expanded to partition the full image "
        "(matching Phase 1 behavior) so LoRA strength is preserved."
    )

    def convert(self, freefuse_data, mask_1, latent=None,
                partition_mode="full_partition", generate_background=True,
                **kwargs):
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

        # Build seed masks at latent resolution
        concept_names = []
        seed_masks = []
        for i, adapter_info in enumerate(adapters):
            name = adapter_info.get("name")
            if not name:
                continue
            if i < len(input_masks):
                resized = self._resize_mask(input_masks[i], target_h, target_w)
                concept_names.append(name)
                seed_masks.append(resized)
                print(
                    f"[FreeFuse MaskFromImage] Mapped mask_{i + 1} → "
                    f"'{name}' ({target_h}x{target_w})"
                )
            else:
                print(
                    f"[FreeFuse MaskFromImage] No mask for adapter '{name}', skipping"
                )

        if not seed_masks:
            return ({"masks": {}},)

        if partition_mode == "full_partition":
            mask_dict = self._full_partition(
                concept_names, seed_masks, target_h, target_w,
                generate_background,
            )
        else:
            # seed_only: use masks as-is
            mask_dict = {}
            for name, m in zip(concept_names, seed_masks):
                mask_dict[name] = m
            if generate_background:
                covered = torch.zeros(target_h, target_w)
                for m in seed_masks:
                    covered = torch.max(covered, m)
                mask_dict["_background_"] = (1.0 - covered).clamp(0, 1)

        return ({"masks": mask_dict},)

    def _full_partition(self, concept_names, seed_masks, h, w,
                        generate_background):
        """Expand seed masks to partition the full image, matching Phase 1.

        Phase 1 algorithm (two-step):
        1. Partition image among concepts only (balanced argmax, no background)
        2. Carve out background separately from originally uncovered regions

        We replicate this:
        1. Distance transform among concepts only → Voronoi-like partition
        2. Intersect with foreground mask (union of original seeds) for background
        """
        # Step 1: Partition entire image among concepts only (no background)
        stacked = torch.stack(seed_masks, dim=0)  # (C, H, W)
        distances = self._distance_transform(stacked, h, w)
        assignment = distances.argmin(dim=0)  # (H, W)

        # Step 2: Determine foreground vs background
        # Foreground = union of all original seed regions
        # Everything else was originally uncovered → background
        if generate_background:
            covered = torch.zeros(h, w)
            for m in seed_masks:
                covered = torch.max(covered, m)
            foreground_mask = covered  # 1 where any concept seed exists

        # Build final masks
        mask_dict = {}
        for i, name in enumerate(concept_names):
            concept_mask = (assignment == i).float()
            if generate_background:
                # Only keep concept assignment where foreground exists,
                # but also keep the full partition for the concept's own
                # seed region and its expanded territory
                # Phase 1 approach: concept gets full partition, then
                # background is overlaid. But background in Phase 1 is
                # typically small (controlled by bg_scale).
                # For custom masks: skip background carving to match the
                # "divide into two areas" case that works well.
                pass
            mask_dict[name] = concept_mask
            coverage = concept_mask.sum() / (h * w) * 100
            print(f"[FreeFuse MaskFromImage] '{name}': {coverage:.1f}% coverage")

        # Add background: originally uncovered pixels keep their
        # concept assignment but we also provide a background mask
        # for the preview node. The background mask is intentionally
        # empty (all zeros) since concepts should own the full image
        # for strong LoRA effects — matching Phase 1 where bg_scale
        # controls a typically small background region.
        if generate_background:
            mask_dict["_background_"] = torch.zeros(h, w)
            print(f"[FreeFuse MaskFromImage] '_background_': 0.0% coverage "
                  f"(concepts partition full image)")

        return mask_dict

    def _distance_transform(self, seeds, h, w):
        """Approximate distance transform via iterative min-pool.

        For each concept channel, returns a (H, W) tensor where each pixel
        holds the approximate distance to the nearest seed pixel.
        Pixels inside the seed region have distance 0.
        """
        C = seeds.shape[0]
        max_dist = float(h + w)
        distances = torch.where(
            seeds > 0.5,
            torch.zeros(C, h, w),
            torch.full((C, h, w), max_dist),
        )

        # Iterative propagation: each iteration extends by 1 pixel
        num_iters = h + w  # ensure full coverage for worst case
        dist = distances.unsqueeze(0)  # (1, C, H, W)
        for _ in range(num_iters):
            pooled = -F.max_pool2d(
                -dist, kernel_size=3, stride=1, padding=1
            )
            dist = torch.min(dist, pooled + 1.0)
            if (dist < max_dist).all():
                break

        return dist.squeeze(0)  # (C, H, W)

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
