"""
01_validate_stack.py — Run drift/quality checks on image stacks.

Usage:
    python scripts/01_validate_stack.py [--dataset MAIN-SET1|leaf2|all]

Runs Step 1 checks (global energy, local energy, feature tracking)
and reports whether frame registration is needed.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR
from src.loader import load_stack
from src.drift_check import run_drift_check


def main():
    parser = argparse.ArgumentParser(description="Validate focus stack quality")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset name or 'all' (default: all)")
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    else:
        datasets = [DATA_DIR / args.dataset]

    if not datasets:
        print(f"No datasets found in {DATA_DIR}")
        print("Run 00_extract_data.py first.")
        sys.exit(1)

    results = {}

    for ds_path in datasets:
        if not ds_path.is_dir():
            print(f"⚠ Dataset directory not found: {ds_path}")
            continue

        name = ds_path.name
        print(f"\n{'='*60}")
        print(f"Validating: {name}")
        print(f"{'='*60}")

        stack, metadata = load_stack(ds_path)

        result = run_drift_check(stack, name)
        results[name] = result

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for name, result in results.items():
        status = "DRIFT DETECTED" if result["drift_detected"] else "CLEAN"
        print(f"  {name}: {status} (max drift = {result['max_drift_px']:.3f} px)")

    needs_registration = any(r["drift_detected"] for r in results.values())
    if needs_registration:
        print("\n→ Frame registration (Step 2) is needed. Run 02_register_frames.py")
    else:
        print("\n→ No registration needed. Proceed to 03_run_pipeline.py")


if __name__ == "__main__":
    main()
