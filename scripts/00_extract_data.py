"""
00_extract_data.py — Extract focus-stack image sets from zip archives.

Usage:
    python scripts/00_extract_data.py

Extracts MAIN-SET1 and leaf2 zip files into data/ directory,
verifies frame counts, and prints image metadata.
"""

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, PROJECT_ROOT as PROJ


def extract_dataset(zip_path: Path, target_name: str) -> Path:
    """
    Extract a zip file into data/<target_name>/.

    WHY: We extract into a named directory rather than letting the zip's
    internal structure dictate layout, so the rest of the pipeline has
    a predictable path to find frames.
    """
    target_dir = DATA_DIR / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting: {zip_path.name}")
    print(f"  → Target: {target_dir}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        print(f"  → Archive contains {len(members)} entries")

        extracted_count = 0
        for member in members:
            if member.endswith('/') or Path(member).name.startswith('.'):
                continue

            filename = Path(member).name
            target_file = target_dir / filename

            with zf.open(member) as src, open(target_file, 'wb') as dst:
                dst.write(src.read())
            extracted_count += 1

        print(f"  → Extracted {extracted_count} files")

    return target_dir


def verify_dataset(directory: Path, expected_name: str) -> None:
    """Verify the extracted dataset has images and print basic info."""
    extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
    image_files = sorted(
        f for f in directory.iterdir()
        if f.suffix.lower() in extensions
    )

    if not image_files:
        print(f"  ⚠ WARNING: No image files found in {directory}!")
        print(f"    Files present: {[f.name for f in directory.iterdir()][:10]}")
        return

    print(f"\n  Dataset: {expected_name}")
    print(f"  Frame count: {len(image_files)}")
    print(f"  Format: {image_files[0].suffix}")
    print(f"  First frame: {image_files[0].name}")
    print(f"  Last frame:  {image_files[-1].name}")

    import cv2
    sample = cv2.imread(str(image_files[0]), cv2.IMREAD_UNCHANGED)
    if sample is not None:
        print(f"  Dimensions: {sample.shape[1]}×{sample.shape[0]} pixels")
        print(f"  Dtype: {sample.dtype}")
        print(f"  Channels: {sample.shape[2] if sample.ndim == 3 else 1}")
    else:
        print(f"  ⚠ Could not read first image: {image_files[0]}")


def main():
    print("=" * 60)
    print("Shape-from-Focus — Data Extraction")
    print("=" * 60)

    zip_files = list(PROJ.glob("*.zip"))

    if not zip_files:
        print("ERROR: No zip files found in project root!")
        print(f"  Looked in: {PROJ}")
        sys.exit(1)

    print(f"\nFound {len(zip_files)} zip file(s):")
    for zf in zip_files:
        print(f"  • {zf.name} ({zf.stat().st_size / 1e6:.1f} MB)")

    datasets = {}
    for zf in zip_files:
        name = zf.stem.split('-20')[0]  # Strip timestamp suffix
        target_dir = extract_dataset(zf, name)
        datasets[name] = target_dir

    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    for name, directory in datasets.items():
        verify_dataset(directory, name)

    print("\n" + "=" * 60)
    print("Data extraction complete.")
    print(f"Data directory: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
