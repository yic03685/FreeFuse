"""
FreeFuse Mask From Image

Convert user-provided masks (ComfyUI MASK type) into FREEFUSE_MASKS,
bypassing Phase 1 entirely. Each mask corresponds to one LoRA concept.

Phase 1 masks partition the ENTIRE image — every pixel belongs to exactly
one concept. This node replicates that behavior: user-provided seed masks
mark where each concept is anchored, and uncovered pixels are assigned to
the nearest concept via distance transform, producing a full partition.
"""

import torch
import torch.nn.functional as F


class FreeFuseMaskFromImage:
    """
    Convert per-concept pixel masks into FREEFUSE_MASKS for Phase 2.

    Provide one MASK per LoRA concept. The node reads adapter names from
    freefuse_data and maps them in order: mask_1 → first adapter, etc.
    Masks are resized to latent resolution automatically.

    By default, masks are expanded to cover the full image (like Phase 1),
    assigning uncovered pixels to the nearest concept. This prevents
    weak LoRA effects from small/tight masks.
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
                    "Include a background concept. In full_partition mode, "
                    "uncovered pixels become background seeds before expansion. "
                    "In seed_only mode, uncovered pixels get a background mask."
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
                mask_dict["background"] = (1.0 - covered).clamp(0, 1)

        return ({"masks": mask_dict},)

    def _full_partition(self, concept_names, seed_masks, h, w,
                        generate_background):
        """Expand seed masks to partition the full image via nearest-concept
        assignment, matching Phase 1 behavior.

        Algorithm:
        1. Stack all seed masks (+ optional background for uncovered pixels).
        2. For each concept, compute a distance transform from its seed region.
        3. Assign every pixel to the concept with the smallest distance.
        4. Result: every pixel belongs to exactly one concept.
        """
        num_concepts = len(seed_masks)

        # Build seed tensor: (C, H, W)
        all_names = list(concept_names)
        all_seeds = list(seed_masks)

        if generate_background:
            covered = torch.zeros(h, w)
            for m in seed_masks:
                covered = torch.max(covered, m)
            bg_seed = (1.0 - covered).clamp(0, 1)
            # Only add background if there are uncovered pixels
            if bg_seed.sum() > 0:
                all_names.append("background")
                all_seeds.append(bg_seed)

        C = len(all_names)
        stacked = torch.stack(all_seeds, dim=0)  # (C, H, W)

        # Distance transform: for each concept, compute distance from seed
        # Use iterative dilation as a GPU-friendly approximation
        distances = self._distance_transform(stacked, h, w)

        # Assign each pixel to the nearest concept (smallest distance)
        assignment = distances.argmin(dim=0)  # (H, W)

        # Build final masks
        mask_dict = {}
        for i, name in enumerate(all_names):
            mask_dict[name] = (assignment == i).float()
            coverage = mask_dict[name].sum() / (h * w) * 100
            print(f"[FreeFuse MaskFromImage] '{name}': {coverage:.1f}% coverage")

        return mask_dict

    def _distance_transform(self, seeds, h, w):
        """Approximate distance transform via iterative max-pool erosion.

        For each concept channel, returns a (H, W) tensor where each pixel
        holds the approximate distance to the nearest seed pixel.
        Pixels inside the seed region have distance 0.
        """
        C = seeds.shape[0]
        # Large initial distance for non-seed pixels
        max_dist = float(h + w)
        distances = torch.where(
            seeds > 0.5,
            torch.zeros(C, h, w),
            torch.full((C, h, w), max_dist),
        )

        # Iterative propagation: each iteration extends by 1 pixel
        # Using min-pool (= -max_pool(-x)) to propagate minimum distances
        num_iters = max(h, w)
        dist = distances.unsqueeze(0)  # (1, C, H, W)
        for _ in range(num_iters):
            # Min-pool with 3x3 kernel propagates distance by 1 pixel
            pooled = -F.max_pool2d(
                -dist, kernel_size=3, stride=1, padding=1
            )
            dist = torch.min(dist, pooled + 1.0)
            # Early exit if fully propagated
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
