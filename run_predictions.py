# run_predictions.py

"""
Main script to run all prediction models (regression and classification).

This script:
1. Builds the player snapshot from raw data
2. Runs pretrained regression model for market value growth
3. Runs pretrained classification model for breakout prediction

Usage:
    python run_predictions.py
    
    # Or run only specific models:
    python run_predictions.py --regression-only
    python run_predictions.py --classification-only
"""

import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction pipeline for player market value and breakout analysis"
    )
    parser.add_argument(
        "--regression-only",
        action="store_true",
        help="Run only regression model"
    )
    parser.add_argument(
        "--classification-only",
        action="store_true",
        help="Run only classification model"
    )
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="Skip building player snapshot (use existing)"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    python_exe = sys.executable

    # Determine which models to run
    run_regression = not args.classification_only
    run_classification = not args.regression_only

    # ----------------- Step 1: build player snapshot -----------------
    if not args.skip_snapshot:
        snapshot_script = project_root / "src" / "data_helper" / "build_player_snapshot.py"
        print(f"[run_predictions] Step 1/3: Building player snapshot...")
        print(f"[run_predictions] Running: {python_exe} {snapshot_script}")

        try:
            subprocess.run([python_exe, str(snapshot_script)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[run_predictions] ERROR: Failed to build player snapshot: {e}")
            sys.exit(1)
    else:
        print("[run_predictions] Skipping player snapshot build (--skip-snapshot flag)")

    # ----------------- Step 2: run pretrained regression ------------- 
    if run_regression:
        print("\n[run_predictions] Step 2/3: Running pretrained regression model...")
        print("[run_predictions] Predicting market value growth...")

        try:
            # Import here so that Python sees the src package
            from src.models.regression.regression_pretrained import run_pretrained as run_regression_pretrained

            run_regression_pretrained()
            print("[run_predictions] ✓ Regression predictions complete")
        except Exception as e:
            print(f"[run_predictions] ERROR: Regression failed: {e}")
            if not run_classification:
                sys.exit(1)
    else:
        print("\n[run_predictions] Skipping regression model")

    # ----------------- Step 3: run pretrained classification ---------
    if run_classification:
        print("\n[run_predictions] Step 3/3: Running pretrained classification model...")
        print("[run_predictions] Predicting player breakout probability...")

        try:
            # Import classification runner
            from src.models.classification.classification_pretrained import run_pretrained as run_classification_pretrained

            run_classification_pretrained()
            print("[run_predictions] ✓ Classification predictions complete")
        except Exception as e:
            print(f"[run_predictions] ERROR: Classification failed: {e}")
            sys.exit(1)
    else:
        print("\n[run_predictions] Skipping classification model")

    # ----------------- Summary -----------------
    print("\n" + "="*60)
    print("[run_predictions] All predictions completed successfully!")
    print("="*60)
    
    if run_regression:
        print("\n📊 Regression outputs:")
        print("  - data/processed/regression_outputs.parquet")
        print("  - data/processed/player_predictions.csv")
    
    if run_classification:
        print("\n🎯 Classification outputs:")
        print("  - data/processed/classification_outputs.parquet")
        print("  - data/processed/breakout_predictions.csv")
    


if __name__ == "__main__":
    main()