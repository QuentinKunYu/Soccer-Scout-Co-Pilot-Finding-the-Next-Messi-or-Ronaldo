# run_predictions.py

import sys
import subprocess
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    python_exe = sys.executable

    # ----------------- Step 1: build player snapshot -----------------
    snapshot_script = project_root / "src" / "data_helper" / "build_player_snapshot.py"
    print(f"[run_predictions] Step 1/2: Building player snapshot...")
    print(f"[run_predictions] Running: {python_exe} {snapshot_script}")

    subprocess.run([python_exe, str(snapshot_script)], check=True)

    # ----------------- Step 2: run pretrained regression ------------- 
    print("[run_predictions] Step 2/2: Running pretrained regression model...")

    # Import here so that Python sees the src package with project_root on sys.path
    from src.models.regression.regression_pretrained import run_pretrained

    run_pretrained()
    print("[run_predictions] All done")


if __name__ == "__main__":
    main()
