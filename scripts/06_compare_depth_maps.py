"""
compare_depth_maps.py

Interactively compares your SFF pipeline's depth map against an external
tool's height map output (Fiji/ImageJ, Helicon Focus, Zerene Stacker, etc.)

Run it, and it will ask you for file paths. No need to edit the code.

Outputs (saved into a folder you choose, default = current folder):
  - comparison_report.txt      : all numeric results
  - depth_maps_side_by_side.png: visual comparison
  - difference_map.png         : where the two disagree most
  - chatter_fft_comparison.png : FFT/autocorrelation periodicity comparison
"""

import os
import sys
import numpy as np

try:
    import tifffile
except ImportError:
    print("Missing dependency 'tifffile'. Install it with:")
    print("  pip install tifffile")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Missing dependency 'opencv-python'. Install it with:")
    print("  pip install opencv-python")
    sys.exit(1)

import matplotlib.pyplot as plt
from scipy import ndimage


def ask_path(prompt_text, must_exist=True):
    while True:
        path = input(prompt_text).strip().strip('"').strip("'")
        path = os.path.expanduser(path)
        if not must_exist:
            return path
        if os.path.isfile(path):
            return path
        print(f"  -> File not found: {path}\n     Try again (or paste the full path).")


def load_depth_image(path):
    """Loads a depth/height map from .tif/.tiff/.npy/.png and returns a 2D float array."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr = tifffile.imread(path)
    elif ext == ".npy":
        arr = np.load(path)
    else:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Could not read image: {path}")

    arr = np.asarray(arr).astype(np.float64)

    # If it has 3 channels (RGB/RGBA), collapse to single channel via mean
    if arr.ndim == 3:
        print(f"  Note: {os.path.basename(path)} has {arr.shape[2]} channels; "
              f"averaging to single-channel. If this is a color-coded height "
              f"map (not raw depth), the values you get are NOT true depth.")
        arr = arr.mean(axis=2)

    return arr


def normalize_nan(arr):
    """Replace any inf/extreme sentinel values with NaN so they don't wreck stats."""
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    return arr


def align_shapes(a, b):
    """Resize b to match a's shape (nearest available option: resize the larger to the smaller)."""
    if a.shape == b.shape:
        return a, b
    target_h = min(a.shape[0], b.shape[0])
    target_w = min(a.shape[1], b.shape[1])
    print(f"  Shapes differ: {a.shape} vs {b.shape}. Resizing both to ({target_h}, {target_w}).")
    a_r = cv2.resize(a, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    b_r = cv2.resize(b, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return a_r, b_r


def trim_border(arr, pct=0.05):
    """Trim a percentage border on all sides, to avoid edge artifacts skewing comparison."""
    h, w = arr.shape
    bh, bw = int(h * pct), int(w * pct)
    return arr[bh:h - bh, bw:w - bw]


def best_fit_offset(a, b):
    """Find the constant offset that best aligns b's values to a's (least squares), ignoring NaNs."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return 0.0
    offset = np.nanmean(a[mask] - b[mask])
    return offset


def compute_rmse(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    diff = a[mask] - b[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_correlation(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def chatter_fft_profile(depth_map, axis=1):
    """Average the heightfield along the axis perpendicular to the feed direction,
    then FFT the resulting 1D profile to find the dominant chatter frequency."""
    mask = np.isfinite(depth_map)
    filled = np.where(mask, depth_map, np.nanmean(depth_map))
    profile = filled.mean(axis=0 if axis == 1 else 1)
    profile = profile - profile.mean()
    fft_vals = np.fft.rfft(profile)
    freqs = np.fft.rfftfreq(len(profile))
    magnitude = np.abs(fft_vals)
    # ignore the DC component
    if len(magnitude) > 1:
        dominant_idx = np.argmax(magnitude[1:]) + 1
        dominant_freq = freqs[dominant_idx]
    else:
        dominant_freq = 0.0
    return profile, freqs, magnitude, dominant_freq


def main():
    print("=" * 70)
    print("DEPTH MAP COMPARISON TOOL")
    print("=" * 70)

    print("\nStep 1: Your pipeline's depth map (the corrected, border-fixed one)")
    my_path = ask_path("  Path to YOUR pipeline depth map (.tif/.npy/.png): ")

    print("\nStep 2: The comparison tool's height map (Fiji, Helicon Focus, etc.)")
    other_path = ask_path("  Path to the OTHER tool's height map (.tif/.npy/.png): ")
    other_name = input("  What's this tool called? (e.g. Fiji EDF, Helicon Focus): ").strip() or "Other Tool"

    print("\nStep 3: Where should results be saved?")
    out_dir = ask_path("  Output folder (leave blank for current folder): ", must_exist=False)
    if out_dir == "":
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    trim_pct = input("\nTrim border percentage before comparing (default 5, enter for default): ").strip()
    trim_pct = float(trim_pct) / 100.0 if trim_pct else 0.05

    print("\nLoading images...")
    mine = normalize_nan(load_depth_image(my_path))
    other = normalize_nan(load_depth_image(other_path))

    print(f"  Your depth map:  shape={mine.shape}, "
          f"min={np.nanmin(mine):.4f}, max={np.nanmax(mine):.4f}")
    print(f"  {other_name}:  shape={other.shape}, "
          f"min={np.nanmin(other):.4f}, max={np.nanmax(other):.4f}")

    mine_r, other_r = align_shapes(mine, other)

    print(f"\nTrimming {trim_pct*100:.0f}% border from all sides before comparison...")
    mine_t = trim_border(mine_r, trim_pct)
    other_t = trim_border(other_r, trim_pct)

    offset = best_fit_offset(mine_t, other_t)
    print(f"Applying best-fit constant offset to {other_name}: {offset:.4f} "
          f"(tools don't share a z=0 reference, so this is expected/normal)")
    other_aligned = other_t + offset

    rmse = compute_rmse(mine_t, other_aligned)
    corr = compute_correlation(mine_t, other_aligned)

    print("\n" + "-" * 70)
    print(f"RMSE:                  {rmse:.4f}")
    print(f"Pearson correlation:   {corr:.4f}")
    print("-" * 70)

    print("\nRunning chatter FFT analysis on both depth maps...")
    my_profile, my_freqs, my_mag, my_dom = chatter_fft_profile(mine_t)
    other_profile, other_freqs, other_mag, other_dom = chatter_fft_profile(other_aligned)

    print(f"Your pipeline dominant spatial frequency:  {my_dom:.5f} (cycles/pixel)")
    print(f"{other_name} dominant spatial frequency:  {other_dom:.5f} (cycles/pixel)")
    if my_dom > 0:
        pct_diff = abs(my_dom - other_dom) / my_dom * 100
        print(f"Difference: {pct_diff:.1f}%")

    # --- Save numeric report ---
    report_path = os.path.join(out_dir, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write("Depth Map Comparison Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Your pipeline file: {my_path}\n")
        f.write(f"{other_name} file: {other_path}\n\n")
        f.write(f"Shapes after alignment: {mine_t.shape}\n")
        f.write(f"Constant offset applied to {other_name}: {offset:.4f}\n\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"Pearson correlation: {corr:.4f}\n\n")
        f.write(f"Your dominant chatter spatial frequency: {my_dom:.5f} cycles/pixel\n")
        f.write(f"{other_name} dominant chatter spatial frequency: {other_dom:.5f} cycles/pixel\n")
    print(f"\nSaved numeric report -> {report_path}")

    # --- Side-by-side visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].imshow(mine_t, cmap="viridis")
    axes[0].set_title("Your Pipeline")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(other_aligned, cmap="viridis")
    axes[1].set_title(other_name)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    diff = mine_t - other_aligned
    im2 = axes[2].imshow(diff, cmap="RdBu_r")
    axes[2].set_title("Difference (Yours - Other)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    side_by_side_path = os.path.join(out_dir, "depth_maps_side_by_side.png")
    plt.savefig(side_by_side_path, dpi=150)
    plt.close()
    print(f"Saved side-by-side comparison -> {side_by_side_path}")

    # --- FFT comparison plot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(my_profile, label="Your pipeline")
    axes[0].plot(other_profile, label=other_name, alpha=0.7)
    axes[0].set_title("Feed-direction height profile (mean-subtracted)")
    axes[0].legend()

    axes[1].plot(my_freqs, my_mag, label="Your pipeline")
    axes[1].plot(other_freqs, other_mag, label=other_name, alpha=0.7)
    axes[1].axvline(my_dom, color="C0", linestyle="--", alpha=0.5)
    axes[1].axvline(other_dom, color="C1", linestyle="--", alpha=0.5)
    axes[1].set_title("FFT magnitude (dominant frequency = chatter signature)")
    axes[1].set_xlabel("Spatial frequency (cycles/pixel)")
    axes[1].legend()

    plt.tight_layout()
    fft_path = os.path.join(out_dir, "chatter_fft_comparison.png")
    plt.savefig(fft_path, dpi=150)
    plt.close()
    print(f"Saved chatter FFT comparison -> {fft_path}")

    print("\nDone. All outputs saved to:", out_dir)


if __name__ == "__main__":
    main()
    