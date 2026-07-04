"""
confidence.py — Per-pixel confidence scoring for depth estimates.

Step 6 of the pipeline: Quantify how reliable each pixel's depth estimate is.
Without this, the output presents texture-rich cavity regions and flat
background regions as equally trustworthy, which is scientifically dishonest.

Three confidence signals, combined multiplicatively:

1. Peak prominence: How far does the focus-measure peak rise above the
   background? Low prominence → flat, textureless region → no real signal.

2. Fit quality (R²): How well does the Gaussian fit match the actual
   focus-measure curve? Poor fit → depth estimate is uncertain.

3. Edge penalty: Is the peak at the first or last frame? If so, the
   surface may extend beyond the scan range → depth is clipped, not measured.

Combined: C = prominence × R² × edge_penalty ∈ [0, 1]

REF: No single paper for this exact formulation — it's a standard practice
in SFF to report confidence. Nayar & Nakagawa 1994 §4.2 discuss focus-measure
reliability. Pertuz et al. 2013 evaluate "confidence measures".
"""

import numpy as np

from .config import CONFIDENCE_CFG


def compute_confidence(
    focus_volume: np.ndarray,
    r_squared: np.ndarray,
    fit_valid: np.ndarray,
    config: object = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pixel confidence score.

    Parameters
    ----------
    focus_volume : np.ndarray, shape (N, H, W)
        Full focus measure volume.
    r_squared : np.ndarray, shape (H, W)
        Gaussian fit R² from depth estimation.
    fit_valid : np.ndarray, shape (H, W), bool
        Whether Gaussian fit was valid (concave).

    Returns
    -------
    confidence : np.ndarray, shape (H, W), float32
        Combined confidence score in [0, 1].
    mask : np.ndarray, shape (H, W), bool
        True for pixels above the confidence threshold (reliable).
    """
    if config is None:
        config = CONFIDENCE_CFG

    n_frames, h, w = focus_volume.shape
    confidence = np.ones((h, w), dtype=np.float32)

    fm_max = np.max(focus_volume, axis=0)       # (H, W)
    fm_median = np.median(focus_volume, axis=0)  # (H, W)

    prominence = (fm_max - fm_median) / (fm_median + 1e-10)

    prom_norm = np.tanh(prominence / config.min_prominence_ratio - 1.0)
    prom_norm = np.clip(prom_norm, 0, 1)

    confidence *= prom_norm

    r2_score = r_squared.copy()
    r2_score[~fit_valid] = 0.0  # Invalid fits get zero
    r2_score = np.clip(r2_score, 0, 1)

    confidence *= (r2_score ** config.r_squared_weight)

    peak_frames = np.argmax(focus_volume, axis=0)  # (H, W)

    edge_penalty = np.ones((h, w), dtype=np.float32)
    edge_penalty[peak_frames == 0] = 0.2
    edge_penalty[peak_frames == n_frames - 1] = 0.2

    confidence *= edge_penalty

    mask = confidence >= config.confidence_threshold

    n_total = h * w
    n_reliable = np.sum(mask)
    print(f"Confidence scoring:")
    print(f"  Reliable pixels: {n_reliable}/{n_total} ({100*n_reliable/n_total:.1f}%)")
    print(f"  Confidence range: [{confidence.min():.4f}, {confidence.max():.4f}]")
    print(f"  Mean confidence (reliable): {confidence[mask].mean():.4f}" if n_reliable > 0 else "")

    return confidence, mask
