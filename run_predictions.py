"""
Main script to run complete prediction pipeline.

This script:
1. Builds the player snapshot from raw data
2. Builds player development dataset (aging / curves)
3. Runs pretrained regression model for market value growth
4. Runs pretrained classification model for breakout prediction
5. Generates final player recommendations by combining all outputs

Usage:
    python run_predictions.py
    
    # Or run only specific models:
    python run_predictions.py --regression-only
    python run_predictions.py --classification-only
    python run_predictions.py --recommendations-only
    
    # Skip steps:
    python run_predictions.py --skip-snapshot
    python run_predictions.py --skip-development
"""

import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run complete prediction pipeline for player analysis"
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
        "--recommendations-only",
        action="store_true",
        help="Run only recommendations generation (requires existing model outputs)"
    )
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="Skip building player snapshot (use existing)"
    )
    parser.add_argument(
        "--skip-development",
        action="store_true",
        help="Skip building development dataset (use existing)"
    )
    parser.add_argument(
        "--skip-recommendations",
        action="store_true",
        help="Skip recommendations generation"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    python_exe = sys.executable

    # Determine which models / steps to run
    if args.recommendations_only:
        run_regression = False
        run_classification = False
        run_recommendations = True
        run_development = False
    else:
        run_regression = not args.classification_only
        run_classification = not args.regression_only
        run_recommendations = not args.skip_recommendations
        run_development = not args.skip_development

    print("\n" + "=" * 70)
    print("PLAYER PREDICTION PIPELINE")
    print("=" * 70)
    
    # Build list of steps for pretty printing + step counts
    steps_to_run = []
    if not args.skip_snapshot and not args.recommendations_only:
        steps_to_run.append("Build Player Snapshot")
    if run_development and not args.recommendations_only:
        steps_to_run.append("Build Player Development Dataset")
    if run_regression:
        steps_to_run.append("Regression Model (Market Value)")
    if run_classification:
        steps_to_run.append("Classification Model (Breakout)")
    if run_recommendations:
        steps_to_run.append("Generate Recommendations")

    total_steps = len(steps_to_run)

    print(f"\nPipeline steps to execute ({total_steps}):")
    for i, step in enumerate(steps_to_run, 1):
        print(f"  {i}. {step}")
    print()

    # Helper to get step index for pretty headers
    def step_idx(name: str) -> int:
        return steps_to_run.index(name) + 1

    # ----------------- Step 1: Build player snapshot -----------------
    if not args.skip_snapshot and not args.recommendations_only:
        header_step = "Build Player Snapshot"
        print("=" * 70)
        print(f"STEP {step_idx(header_step)}/{total_steps}: {header_step.upper()}")
        print("=" * 70)
        
        snapshot_script = project_root / "src" / "data_helper" / "build_player_snapshot.py"
        print(f"Running: {python_exe} {snapshot_script}\n")

        try:
            subprocess.run([python_exe, str(snapshot_script)], check=True)
            print("\n Player snapshot built successfully")
        except subprocess.CalledProcessError as e:
            print(f"\n ERROR: Failed to build player snapshot: {e}")
            sys.exit(1)
    elif args.skip_snapshot:
        print(" Skipping player snapshot build (--skip-snapshot flag)")

    # ----------------- Step 2: Build development dataset ------------ 
    if run_development and not args.recommendations_only:
        header_step = "Build Player Development Dataset"
        print("\n" + "=" * 70)
        print(f"STEP {step_idx(header_step)}/{total_steps}: {header_step.upper()}")
        print("=" * 70)
        print("Building development curves / aging dataset...\n")

        development_script = project_root / "src" / "data_helper" / "build_development_dataset.py"
        print(f"Running: {python_exe} {development_script}\n")

        try:
            subprocess.run([python_exe, str(development_script)], check=True)
            print("\n Development dataset built successfully")
        except subprocess.CalledProcessError as e:
            print(f"\n ERROR: Failed to build development dataset: {e}")
            sys.exit(1)
    elif args.skip_development and not args.recommendations_only:
        print(" Skipping development dataset build (--skip-development flag)")

    # ----------------- Step 3: Run pretrained regression ------------- 
    if run_regression:
        header_step = "Regression Model (Market Value)"
        print("\n" + "=" * 70)
        print(f"STEP {step_idx(header_step)}/{total_steps}: {header_step.upper()}")
        print("=" * 70)
        print("Predicting 1-year market value growth...\n")

        try:
            from src.models.regression.regression_pretrained import run_pretrained as run_regression_pretrained

            run_regression_pretrained()
            print("\n Regression predictions complete")
        except Exception as e:
            print(f"\n ERROR: Regression failed: {e}")
            import traceback
            traceback.print_exc()
            if not run_classification and not run_recommendations:
                sys.exit(1)
    else:
        print("\n⊳ Skipping regression model")

    # ----------------- Step 4: Run pretrained classification ---------
    if run_classification:
        header_step = "Classification Model (Breakout)"
        print("\n" + "=" * 70)
        print(f"STEP {step_idx(header_step)}/{total_steps}: {header_step.upper()}")
        print("=" * 70)
        print("Predicting player breakout probability...\n")

        try:
            from src.models.classification.classification_pretrained import run_pretrained as run_classification_pretrained

            run_classification_pretrained()
            print("\n Classification predictions complete")
        except Exception as e:
            print(f"\n ERROR: Classification failed: {e}")
            import traceback
            traceback.print_exc()
            if not run_recommendations:
                sys.exit(1)
    else:
        print("\n⊳ Skipping classification model")

    # ----------------- Step 5: Generate recommendations --------------
    if run_recommendations:
        header_step = "Generate Recommendations"
        print("\n" + "=" * 70)
        print(f"STEP {step_idx(header_step)}/{total_steps}: {header_step.upper()}")
        print("=" * 70)
        print("Combining all model outputs into final recommendations...\n")

        try:
            from src.data_helper.generate_recommendations import generate_player_recommendations

            recommendations = generate_player_recommendations()
            print("\n Player recommendations generated successfully")
            print(f"  Total players: {len(recommendations)}")
        except Exception as e:
            print(f"\n ERROR: Recommendations generation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n⊳ Skipping recommendations generation")

    # ----------------- Summary -----------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    print("\n Output Files Generated:")
    print("-" * 70)
    
    if run_development and not args.recommendations_only:
        print("\n Development Dataset:")
        print("  └─ data/processed/development_outputs.parquet")
    
    if run_regression or args.recommendations_only:
        print("\n Regression Model Outputs:")
        print("  ├─ data/processed/regression_outputs.parquet")
        print("  └─ data/processed/player_predictions.csv")
    
    if run_classification or args.recommendations_only:
        print("\n Classification Model Outputs:")
        print("  ├─ data/processed/classification_outputs.parquet")
        print("  └─ data/processed/breakout_predictions.csv")
    
    if run_recommendations:
        print("\n Final Recommendations:")
        print("  ├─ data/processed/player_recommendations.parquet")
        print("  ├─ data/processed/player_recommendations.csv")
        print("  └─ app/mock_data/player_recommendations.csv (for frontend)")
    
    print("\n💡 Next Steps:")
    print("-" * 70)
    
    if run_recommendations:
        print("  ✓ View recommendations:")
        print("    → cat data/processed/player_recommendations.csv | head")
        print("\n  ✓ Load in Python:")
        print("    → import pandas as pd")
        print("    → df = pd.read_csv('data/processed/player_recommendations.csv')")
        print("    → print(df.nlargest(20, 'breakout_prob'))")
        print("\n  ✓ Start frontend app:")
        print("    → cd app && npm start")
    else:
        print("  → Run complete pipeline: python run_predictions.py")
        print("  → Generate recommendations: python run_predictions.py --recommendations-only")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
