#!/usr/bin/env python
"""
Tests for FreeFuseMaskFromImage node.

Verifies that the output matches the Phase 1 (FreeFusePhase1Sampler)
mask format so that the FreeFuseMaskApplicator can consume it correctly.
"""

import importlib.util
import os
import sys
import torch


def _load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_node_class():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    freefuse_dir = os.path.dirname(script_dir)
    path = os.path.join(freefuse_dir, "nodes", "mask_from_image.py")
    mod = _load_module("mask_from_image_test", path)
    return mod.FreeFuseMaskFromImage


FreeFuseMaskFromImage = _get_node_class()


def make_freefuse_data(adapter_names):
    """Create a minimal freefuse_data dict like FreeFuseLoRALoader produces."""
    return {
        "adapters": [{"name": n} for n in adapter_names],
        "token_pos_maps": {n: [[i * 5, i * 5 + 1]] for i, n in enumerate(adapter_names)},
    }


def make_phase1_masks(concept_names, h=64, w=64):
    """Create a reference Phase 1-style mask bank.

    Phase 1 (generate_masks with stabilized_balanced_argmax) produces:
    - Binary masks (0.0 / 1.0)
    - Each pixel assigned to exactly one concept
    - Every concept mask is (H, W)
    - Background mask keyed as '_background_'
    - Full partition: sum of all masks == 1.0 everywhere
    """
    n = len(concept_names)
    # Simulate balanced argmax: divide image into vertical strips
    assignment = torch.zeros(h, w, dtype=torch.long)
    strip_w = w // (n + 1)  # +1 for background
    for i in range(n):
        assignment[:, i * strip_w:(i + 1) * strip_w] = i
    assignment[:, n * strip_w:] = n  # background

    masks = {}
    for i, name in enumerate(concept_names):
        masks[name] = (assignment == i).float()
    masks["_background_"] = (assignment == n).float()

    mask_bank = {
        "masks": masks,
        "similarity_maps": {},
        "metadata": {"phase1_seed": 42},
    }
    return mask_bank


# ──────────────────────────────────────────────────────────
# Format compatibility tests
# ──────────────────────────────────────────────────────────

def test_output_is_dict_with_masks_key():
    """Output must be a dict with 'masks' key (like Phase 1)."""
    node = FreeFuseMaskFromImage()
    freefuse_data = make_freefuse_data(["concept_a", "concept_b"])
    mask_a = torch.ones(1, 512, 512)
    mask_b = torch.zeros(1, 512, 512)

    (result,) = node.convert(freefuse_data, mask_a, mask_2=mask_b)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "masks" in result, f"Missing 'masks' key. Keys: {result.keys()}"
    print("PASS: output is dict with 'masks' key")


def test_mask_keys_match_adapter_names():
    """Mask keys must match adapter names from freefuse_data."""
    node = FreeFuseMaskFromImage()
    names = ["sks_person", "ohwx_dog"]
    freefuse_data = make_freefuse_data(names)
    mask_a = torch.ones(1, 512, 512)
    mask_b = torch.ones(1, 512, 512)

    (result,) = node.convert(freefuse_data, mask_a, mask_2=mask_b)
    mask_dict = result["masks"]

    for name in names:
        assert name in mask_dict, f"Missing mask for adapter '{name}'"
    print(f"PASS: mask keys match adapter names: {list(mask_dict.keys())}")


def test_background_key_format():
    """Background must be keyed as '_background_' (with underscores)
    so the preview node renders it as gray."""
    node = FreeFuseMaskFromImage()
    freefuse_data = make_freefuse_data(["concept_a"])
    mask_a = torch.zeros(1, 512, 512)
    mask_a[:, :, :256] = 1.0  # left half

    (result,) = node.convert(freefuse_data, mask_a)
    mask_dict = result["masks"]

    assert "_background_" in mask_dict, (
        f"Background key should be '_background_'. Keys: {list(mask_dict.keys())}"
    )
    print("PASS: background key is '_background_'")


def test_masks_are_binary():
    """Masks must be binary (0.0 or 1.0) like Phase 1."""
    node = FreeFuseMaskFromImage()
    freefuse_data = make_freefuse_data(["concept_a", "concept_b"])
    mask_a = torch.ones(1, 512, 512) * 0.7  # soft mask
    mask_b = torch.ones(1, 512, 512) * 0.3  # soft mask

    (result,) = node.convert(freefuse_data, mask_a, mask_2=mask_b)
    mask_dict = result["masks"]

    for name, mask in mask_dict.items():
        unique = mask.unique()
        assert all(v in (0.0, 1.0) for v in unique.tolist()), (
            f"Mask '{name}' has non-binary values: {unique.tolist()}"
        )
    print("PASS: all masks are binary (0.0 / 1.0)")


def test_mask_shape_is_2d():
    """Each mask must be (H, W) — 2D tensor at latent resolution."""
    node = FreeFuseMaskFromImage()
    freefuse_data = make_freefuse_data(["concept_a"])
    mask_a = torch.ones(1, 1024, 1024)

    (result,) = node.convert(freefuse_data, mask_a)
    mask_dict = result["masks"]

    for name, mask in mask_dict.items():
        assert mask.dim() == 2, (
            f"Mask '{name}' should be 2D, got {mask.dim()}D shape {mask.shape}"
        )
    print("PASS: all masks are 2D (H, W)")


def test_mask_at_latent_resolution():
    """Masks must be downscaled to latent resolution (not pixel resolution)."""
    node = FreeFuseMaskFromImage()
    freefuse_data = make_freefuse_data(["concept_a"])
    mask_a = torch.ones(1, 1024, 1024)

    (result,) = node.convert(freefuse_data, mask_a)
    mask_dict = result["masks"]

    for name, mask in mask_dict.items():
        assert mask.shape[0] <= 128 and mask.shape[1] <= 128, (
            f"Mask '{name}' shape {mask.shape} looks like pixel resolution, "
            f"expected latent resolution (e.g. 64x64 for 1024x1024)"
        )
    print(f"PASS: masks at latent resolution: {list(mask_dict.values())[0].shape}")


def test_latent_override():
    """When a latent is provided, mask dims should match latent spatial dims."""
    node = FreeFuseMaskFromImage()
    freefuse_data = make_freefuse_data(["concept_a"])
    mask_a = torch.ones(1, 1024, 768)
    latent = {"samples": torch.zeros(1, 16, 96, 72)}  # non-square

    (result,) = node.convert(freefuse_data, mask_a, latent=latent)
    mask = result["masks"]["concept_a"]

    assert mask.shape == (96, 72), (
        f"Mask shape {mask.shape} doesn't match latent spatial dims (96, 72)"
    )
    print(f"PASS: mask matches latent dims: {mask.shape}")


# ──────────────────────────────────────────────────────────
# Functional tests: mask content
# ──────────────────────────────────────────────────────────

def test_white_means_active():
    """White (1.0) in input mask → 1.0 in output (LoRA active)."""
    node = FreeFuseMaskFromImage()
    freefuse_data = make_freefuse_data(["concept_a"])
    # Full white mask
    mask_a = torch.ones(1, 512, 512)

    (result,) = node.convert(freefuse_data, mask_a)
    concept_mask = result["masks"]["concept_a"]

    assert concept_mask.sum() == concept_mask.numel(), (
        "Full white input should produce full white output"
    )
    print("PASS: white input → active output")


def test_black_means_inactive():
    """Black (0.0) in input mask → 0.0 in output (LoRA inactive)."""
    node = FreeFuseMaskFromImage()
    freefuse_data = make_freefuse_data(["concept_a"])
    mask_a = torch.zeros(1, 512, 512)

    (result,) = node.convert(freefuse_data, mask_a)
    concept_mask = result["masks"]["concept_a"]

    assert concept_mask.sum() == 0, (
        "Full black input should produce full black output"
    )
    print("PASS: black input → inactive output")


def test_background_is_inverse_of_concepts():
    """Background should cover exactly the pixels not covered by any concept."""
    node = FreeFuseMaskFromImage()
    freefuse_data = make_freefuse_data(["concept_a", "concept_b"])

    # Create non-overlapping masks: left third, right third
    mask_a = torch.zeros(1, 512, 512)
    mask_a[:, :, :170] = 1.0
    mask_b = torch.zeros(1, 512, 512)
    mask_b[:, :, 342:] = 1.0

    (result,) = node.convert(freefuse_data, mask_a, mask_2=mask_b)
    mask_dict = result["masks"]

    concept_union = torch.max(mask_dict["concept_a"], mask_dict["concept_b"])
    bg = mask_dict["_background_"]

    # background + concept_union should cover everything
    total = torch.max(concept_union, bg)
    assert total.sum() == total.numel(), (
        "Concept union + background should cover all pixels"
    )
    # No overlap between concepts and background
    overlap = (concept_union * bg).sum()
    assert overlap == 0, (
        f"Background overlaps with concepts: {overlap} pixels"
    )
    print("PASS: background is inverse of concept union")


def test_no_overlap_between_concept_masks():
    """If input masks don't overlap, output masks shouldn't overlap either."""
    node = FreeFuseMaskFromImage()
    freefuse_data = make_freefuse_data(["concept_a", "concept_b"])

    mask_a = torch.zeros(1, 512, 512)
    mask_a[:, :, :256] = 1.0
    mask_b = torch.zeros(1, 512, 512)
    mask_b[:, :, 256:] = 1.0

    (result,) = node.convert(freefuse_data, mask_a, mask_2=mask_b)
    mask_dict = result["masks"]

    overlap = (mask_dict["concept_a"] * mask_dict["concept_b"]).sum()
    assert overlap == 0, f"Concepts should not overlap: {overlap} pixels"
    print("PASS: non-overlapping input → non-overlapping output")


# ──────────────────────────────────────────────────────────
# Comparison with Phase 1 format
# ──────────────────────────────────────────────────────────

def test_matches_phase1_format():
    """Full format comparison with Phase 1 output."""
    node = FreeFuseMaskFromImage()
    names = ["sks_person", "ohwx_dog"]
    freefuse_data = make_freefuse_data(names)

    # Simulate: person on left, dog on right
    mask_person = torch.zeros(1, 1024, 1024)
    mask_person[:, :, :512] = 1.0
    mask_dog = torch.zeros(1, 1024, 1024)
    mask_dog[:, :, 512:] = 1.0

    (result,) = node.convert(freefuse_data, mask_person, mask_2=mask_dog)

    # Compare against Phase 1 reference
    phase1 = make_phase1_masks(names, h=64, w=64)
    phase1_masks = phase1["masks"]
    our_masks = result["masks"]

    # Check structure
    errors = []

    # 1. Both have same concept keys
    for name in names:
        if name not in our_masks:
            errors.append(f"Missing concept key '{name}'")

    # 2. Both have background key
    if "_background_" not in our_masks:
        errors.append("Missing '_background_' key")

    # 3. Both produce 2D tensors
    for name, mask in our_masks.items():
        if mask.dim() != 2:
            errors.append(f"'{name}' is {mask.dim()}D, expected 2D")

    # 4. Both produce binary masks
    for name, mask in our_masks.items():
        unique = mask.unique()
        if not all(v in (0.0, 1.0) for v in unique.tolist()):
            errors.append(f"'{name}' non-binary: {unique.tolist()}")

    # 5. Masks form a partition (no pixel unassigned, no pixel double-assigned)
    all_masks = torch.stack([our_masks[n] for n in our_masks], dim=0)
    pixel_sum = all_masks.sum(dim=0)
    # Each pixel should belong to exactly one mask
    if not torch.allclose(pixel_sum, torch.ones_like(pixel_sum)):
        min_val = pixel_sum.min().item()
        max_val = pixel_sum.max().item()
        errors.append(
            f"Not a valid partition: pixel assignment sum range [{min_val}, {max_val}], "
            f"expected all 1.0"
        )

    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        raise AssertionError(f"{len(errors)} format mismatches with Phase 1")

    print("PASS: output format matches Phase 1")
    print(f"  Keys: {list(our_masks.keys())}")
    for name, mask in our_masks.items():
        coverage = mask.sum() / mask.numel() * 100
        print(f"  {name}: shape={tuple(mask.shape)}, coverage={coverage:.1f}%")


def test_tight_masks_partition():
    """Tight masks (small coverage) with background should still form a valid partition."""
    node = FreeFuseMaskFromImage()
    freefuse_data = make_freefuse_data(["concept_a", "concept_b"])

    # Small masks: ~10% each, 80% background
    mask_a = torch.zeros(1, 512, 512)
    mask_a[:, 100:200, 100:200] = 1.0  # small square
    mask_b = torch.zeros(1, 512, 512)
    mask_b[:, 100:200, 300:400] = 1.0  # small square

    (result,) = node.convert(freefuse_data, mask_a, mask_2=mask_b)
    mask_dict = result["masks"]

    all_masks = torch.stack(list(mask_dict.values()), dim=0)
    pixel_sum = all_masks.sum(dim=0)

    if not torch.allclose(pixel_sum, torch.ones_like(pixel_sum)):
        min_val = pixel_sum.min().item()
        max_val = pixel_sum.max().item()
        raise AssertionError(
            f"Tight masks don't form valid partition: sum range [{min_val}, {max_val}]"
        )

    for name, mask in mask_dict.items():
        coverage = mask.sum() / mask.numel() * 100
        print(f"  {name}: {coverage:.1f}% coverage")
    print("PASS: tight masks form valid partition")


# ──────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_output_is_dict_with_masks_key,
        test_mask_keys_match_adapter_names,
        test_background_key_format,
        test_masks_are_binary,
        test_mask_shape_is_2d,
        test_mask_at_latent_resolution,
        test_latent_override,
        test_white_means_active,
        test_black_means_inactive,
        test_background_is_inverse_of_concepts,
        test_no_overlap_between_concept_masks,
        test_matches_phase1_format,
        test_tight_masks_partition,
    ]

    passed = 0
    failed = 0
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
