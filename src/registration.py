"""
registration.py — ECC-based frame registration for drift correction.

Step 2 of the pipeline: Align all frames to a reference frame using
Enhanced Correlation Coefficient (ECC) registration. Only needed if
Step 1 detects XY drift > 1 pixel.

WHY ECC over feature-based methods (ORB/SIFT):
In a focus stack, most of the image is blurred in most frames — feature
detectors find very few reliable keypoints in defocused frames. ECC
operates on pixel intensities directly and handles blur gracefully because
it maximizes correlation, which is robust to the low-pass filtering effect
of defocus.

WHY MOTION_TRANSLATION:
A microscope focus stack should only exhibit translational drift (the stage
or sample sliding slightly). Rotation and scaling would indicate a more
serious mechanical problem. We start with translation-only and fall back
to Euclidean (adds rotation) only if translation fails on many frames.

REF: Evangelidis & Psarakis, "Parametric Image Alignment Using Enhanced
     Correlation Coefficient Maximization", IEEE TPAMI 30(10), 2008
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from .config import REGISTRATION_CFG


def align_stack(
    stack: np.ndarray,
    config: Optional[object] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Align all frames to the sharpest reference frame using ECC.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W), float32 [0, 1]
    config : RegistrationConfig, optional

    Returns
    -------
    aligned : np.ndarray, same shape as stack
        Aligned frames. Pixels that shift in from outside the FOV are set to 0.
    info : dict
        Registration metadata:
        - ref_frame: int (reference frame index)
        - warp_matrices: list of 2x3 arrays
        - shifts_x, shifts_y: lists of detected shifts per frame
        - failed_frames: list of frame indices that failed alignment
        - model_used: str
    """
    if config is None:
        config = REGISTRATION_CFG

    n_frames, h, w = stack.shape

    if config.motion_model == "translation":
        warp_mode = cv2.MOTION_TRANSLATION
        warp_init = np.eye(2, 3, dtype=np.float32)
    elif config.motion_model == "euclidean":
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_init = np.eye(2, 3, dtype=np.float32)
    else:
        raise ValueError(f"Unknown motion model: {config.motion_model}")

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.max_iterations,
        config.epsilon,
    )

    energies = np.array([
        np.mean(cv2.Laplacian((stack[i] * 255).astype(np.uint8), cv2.CV_64F) ** 2)
        for i in range(n_frames)
    ])
    ref_idx = int(np.argmax(energies))
    ref_frame = (stack[ref_idx] * 255).astype(np.uint8)

    if verbose:
        print(f"Reference frame: {ref_idx} (highest focus energy)")

    aligned = np.copy(stack)
    warp_matrices = [None] * n_frames
    shifts_x = [0.0] * n_frames
    shifts_y = [0.0] * n_frames
    failed_frames = []

    iterator = tqdm(range(n_frames), desc="Aligning frames", unit="frame") if verbose else range(n_frames)

    for i in iterator:
        if i == ref_idx:
            warp_matrices[i] = np.eye(2, 3, dtype=np.float32)
            continue

        frame = (stack[i] * 255).astype(np.uint8)
        warp_matrix = warp_init.copy()

        try:
            cc, warp_matrix = cv2.findTransformECC(
                ref_frame, frame, warp_matrix, warp_mode, criteria
            )

            aligned_frame = cv2.warpAffine(
                stack[i], warp_matrix, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            aligned[i] = aligned_frame
            warp_matrices[i] = warp_matrix

            shifts_x[i] = float(warp_matrix[0, 2])
            shifts_y[i] = float(warp_matrix[1, 2])

        except cv2.error:
            failed_frames.append(i)
            warp_matrices[i] = np.eye(2, 3, dtype=np.float32)

    model_used = config.motion_model

    failure_rate = len(failed_frames) / n_frames
    if failure_rate > config.failure_threshold and config.fallback_to_euclidean \
            and config.motion_model == "translation":
        if verbose:
            print(f"\n⚠ {len(failed_frames)} frames ({failure_rate:.1%}) failed "
                  f"with translation model. Retrying with euclidean...")

        from copy import copy
        fallback_cfg = copy(config)
        fallback_cfg.motion_model = "euclidean"
        fallback_cfg.fallback_to_euclidean = False
        return align_stack(stack, fallback_cfg, verbose)

    if verbose:
        print(f"\nAlignment complete:")
        print(f"  Model: {model_used}")
        print(f"  Failed frames: {len(failed_frames)}/{n_frames}")
        if shifts_x:
            valid_sx = [s for i, s in enumerate(shifts_x) if i not in failed_frames]
            valid_sy = [s for i, s in enumerate(shifts_y) if i not in failed_frames]
            if valid_sx:
                print(f"  X shift range: [{min(valid_sx):.3f}, {max(valid_sx):.3f}] px")
                print(f"  Y shift range: [{min(valid_sy):.3f}, {max(valid_sy):.3f}] px")

    info = {
        "ref_frame": ref_idx,
        "warp_matrices": warp_matrices,
        "shifts_x": shifts_x,
        "shifts_y": shifts_y,
        "failed_frames": failed_frames,
        "model_used": model_used,
    }

    return aligned, info
