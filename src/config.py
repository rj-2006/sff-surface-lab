"""
config.py — Central configuration for the Shape-from-Focus pipeline.

Every tunable parameter lives here with a WHY comment explaining its value
and a REF/ALTERNATIVE comment where applicable. This is the single source
of truth: no magic numbers anywhere else in the codebase.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
DEPTH_MAP_DIR = OUTPUT_DIR / "depth_maps"
MODEL_DIR = OUTPUT_DIR / "3d_models"

for d in [DATA_DIR, DIAGNOSTICS_DIR, DEPTH_MAP_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


Z_STEP_UM = 1.0


XY_UM_PER_PIXEL: Optional[float] = None


# --- PATCH (2026-07-16) ---
# Manual fallback for border removal in the 3D model (visualization.py,
# plot_3d_surface). The automatic border detection there uses the
# confidence mask + connected-component flood-fill from the image edges,
# which adapts to the actual (often ragged) shape of the unreliable rim
# caused by vignetting / registration crop / edge-of-field defocus.
#
# If that automatic detection still leaves a sliver of border in the 3D
# view (e.g. a thin strip that happened to score as "reliable" despite
# being outside the subject), set this to a nonzero pixel count to also
# force-exclude a fixed-width frame around the image from the 3D model
# ONLY. It has no effect on the 2D depth map, confidence overlay, cross
# sections, or dashboard — those are untouched and still show border data.
# Start at 0 (automatic detection only) and increase only if needed.
BORDER_MARGIN_PX: int = 25


@dataclass
class FocusMeasureConfig:
    """Parameters for focus measure computation."""

    sml_window_size: int = 5

    sml_step_size: int = 1

    sml_threshold: float = 0.0

    tenengrad_ksize: int = 3

    tenengrad_window_size: int = 5

    laplacian_ksize: int = 3

    laplacian_window_size: int = 5


@dataclass
class DepthEstimationConfig:
    """Parameters for sub-frame depth interpolation."""

    interp_half_width: int = 2

    min_focus_measure: float = 1.0

    require_concave_fit: bool = True


@dataclass
class SmoothingConfig:
    """Parameters for edge-aware depth smoothing."""

    method: str = "bilateral"

    bilateral_d: int = 9

    bilateral_sigma_color: float = 25.0

    bilateral_sigma_space: float = 9.0

    guided_radius: int = 8

    guided_eps: float = 100.0


@dataclass
class ConfidenceConfig:
    """Parameters for per-pixel confidence scoring."""

    confidence_threshold: float = 0.3

    min_prominence_ratio: float = 0.1

    r_squared_weight: float = 1.0


@dataclass
class DriftCheckConfig:
    """Parameters for input validation and drift detection."""

    n_features: int = 30

    max_drift_px: float = 1.0

    tile_grid: tuple = (4, 4)

    template_size: int = 31

    search_margin: int = 10


@dataclass
class RegistrationConfig:
    """Parameters for ECC frame registration."""

    motion_model: str = "translation"

    max_iterations: int = 50

    epsilon: float = 1e-3

    fallback_to_euclidean: bool = True

    failure_threshold: float = 0.05


FOCUS_CFG = FocusMeasureConfig()
DEPTH_CFG = DepthEstimationConfig()
SMOOTH_CFG = SmoothingConfig()
CONFIDENCE_CFG = ConfidenceConfig()
DRIFT_CFG = DriftCheckConfig()
REGISTRATION_CFG = RegistrationConfig()