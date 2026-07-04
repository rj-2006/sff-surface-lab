"""
loader.py — Image stack loading and metadata extraction.

Loads a directory of microscope images as a 3D numpy array (Z, H, W) where
Z is the frame/focus index. Handles sorting, grayscale conversion, and
basic sanity checks.
"""

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from .config import Z_STEP_UM


def natural_sort_key(path: Path) -> list:
    """
    Sort filenames naturally so that 'frame_2' comes before 'frame_10'.

    WHY: Simple lexicographic sorting puts '10' before '2'. Natural sort
    splits on numeric boundaries and sorts numbers by value.
    """
    parts = re.split(r'(\d+)', path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def load_stack(
    directory: Path,
    grayscale: bool = True,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """
    Load all images from a directory as a focus stack.

    Parameters
    ----------
    directory : Path
        Directory containing the image frames (TIFF, PNG, JPG, BMP).
    grayscale : bool
        If True, convert to grayscale. Focus measures operate on single-channel.
    max_frames : int, optional
        Load only the first N frames (for quick testing).

    Returns
    -------
    stack : np.ndarray
        3D array of shape (n_frames, height, width), dtype float32, values [0, 1].
        WHY float32: Focus measures involve convolutions and sums that overflow
        uint8. Normalizing to [0,1] makes threshold values transferable across
        different bit-depth sources.
    metadata : dict
        Stack metadata: frame_count, height, width, z_step_um, z_range_um,
        source_dtype, source_files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Stack directory not found: {directory}")

    extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
    files = sorted(
        [f for f in directory.iterdir() if f.suffix.lower() in extensions],
        key=natural_sort_key
    )

    if not files:
        raise FileNotFoundError(
            f"No image files found in {directory}. "
            f"Supported extensions: {extensions}"
        )

    if max_frames is not None:
        files = files[:max_frames]

    sample = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    if sample is None:
        raise IOError(f"Failed to read image: {files[0]}")

    source_dtype = sample.dtype
    h, w = sample.shape[:2]

    n_frames = len(files)
    stack = np.empty((n_frames, h, w), dtype=np.float32)

    print(f"Loading {n_frames} frames from {directory.name} ({w}×{h}, {source_dtype})")

    for i, f in enumerate(tqdm(files, desc="Loading frames", unit="frame")):
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to read frame {i}: {f}")

        if img.shape[:2] != (h, w):
            raise ValueError(
                f"Frame {i} ({f.name}) has shape {img.shape[:2]}, "
                f"expected ({h}, {w}). Stack frames must be same size."
            )

        if grayscale and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3:
            img = img[:, :, 0]

        if source_dtype == np.uint8:
            stack[i] = img.astype(np.float32) / 255.0
        elif source_dtype == np.uint16:
            stack[i] = img.astype(np.float32) / 65535.0
        else:
            stack[i] = img.astype(np.float32)

    metadata = {
        "frame_count": n_frames,
        "height": h,
        "width": w,
        "z_step_um": Z_STEP_UM,
        "z_range_um": (n_frames - 1) * Z_STEP_UM,
        "source_dtype": str(source_dtype),
        "source_dir": str(directory),
        "source_files": [f.name for f in files],
    }

    print(f"  → Stack shape: {stack.shape}")
    print(f"  → Z range: 0 to {metadata['z_range_um']:.1f} µm "
          f"({n_frames} frames × {Z_STEP_UM} µm/step)")
    print(f"  → Value range: [{stack.min():.4f}, {stack.max():.4f}]")

    return stack, metadata


def load_single_frame(filepath: Path, grayscale: bool = True) -> np.ndarray:
    """Load a single image frame, normalized to float32 [0, 1]."""
    img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to read image: {filepath}")

    if grayscale and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    else:
        return img.astype(np.float32)
