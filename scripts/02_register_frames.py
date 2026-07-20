"""
02_register_frames.py — Align frames to correct drift (if detected).

Usage:
    python scripts/02_register_frames.py [--dataset MAIN-SET1|leaf2|all]

Only needed if 01_validate_stack.py detected drift > 1 pixel.
Saves aligned frames to data/<dataset>_aligned/.
"""

import sys
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR
from src.loader import load_stack
from src.registration import align_stack, compute_valid_coverage_mask
from src.drift_check import run_drift_check


def save_aligned_stack(aligned: np.ndarray, save_dir: Path):
    """Save aligned frames as individual images."""
    import cv2
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(aligned.shape[0]):
        frame_uint8 = (aligned[i] * 255).astype(np.uint8)
        cv2.imwrite(str(save_dir / f"frame_{i:04d}.png"), frame_uint8)

    print(f"  Saved {aligned.shape[0]} aligned frames to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Register frames to correct drift")
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--force", action="store_true",
                        help="Run even if drift was not detected")
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = [d for d in DATA_DIR.iterdir()
                    if d.is_dir() and not d.name.endswith("_aligned")]
    else:
        datasets = [DATA_DIR / args.dataset]

    for ds_path in datasets:
        if not ds_path.is_dir():
            print(f"⚠ Dataset not found: {ds_path}")
            continue

        name = ds_path.name
        print(f"\n{'='*60}")
        print(f"Registering: {name}")
        print(f"{'='*60}")

        stack, metadata = load_stack(ds_path)

        print("\nPre-registration drift check...")
        pre_check = run_drift_check(stack, f"{name}_pre_registration")

        if not pre_check["drift_detected"] and not args.force:
            print(f"\n✓ No drift detected. Skipping registration for {name}.")
            print("  Use --force to register anyway.")
            continue

        print("\nAligning frames...")
        aligned, info = align_stack(stack)

        print("\nPost-registration drift check...")
        post_check = run_drift_check(aligned, f"{name}_post_registration")

        print(f"\nDrift before: {pre_check['max_drift_px']:.3f} px")
        print(f"Drift after:  {post_check['max_drift_px']:.3f} px")

        if post_check["max_drift_px"] < pre_check["max_drift_px"]:
            print("✓ Registration reduced drift.")
        else:
            print("⚠ Registration did not reduce drift — check the data.")

        # PATCH (2026-07-18): Derive the registration-warp-boundary region
        # directly from the warp matrices actually used for this dataset,
        # instead of guessing a fixed pixel margin downstream. This is
        # what 03_run_pipeline.py's 3D visualization uses to exclude the
        # warp-artifact border from the 3D model — adapts automatically
        # to however much drift THIS dataset actually had.
        h, w = stack.shape[1:]
        valid_mask, coverage = compute_valid_coverage_mask(
            (h, w), info["warp_matrices"]
        )
        n_invalid = int(np.sum(~valid_mask))
        print(f"\nRegistration coverage: {n_invalid}/{valid_mask.size} pixels "
              f"({100 * n_invalid / valid_mask.size:.1f}%) lack full-stack "
              f"coverage (warp-boundary artifact, all layers combined)")

        aligned_dir = DATA_DIR / f"{name}_aligned"
        save_aligned_stack(aligned, aligned_dir)
        np.save(aligned_dir / "valid_coverage_mask.npy", valid_mask)
        np.save(aligned_dir / "coverage_fraction.npy", coverage)
        print(f"  Registration coverage mask saved to "
              f"{aligned_dir / 'valid_coverage_mask.npy'}")


if __name__ == "__main__":
    main()