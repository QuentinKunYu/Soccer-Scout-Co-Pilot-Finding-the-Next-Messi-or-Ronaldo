import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    df = pd.read_parquet("data/processed/regression_outputs.parquet")

    if "y_growth" not in df.columns:
        raise ValueError("regression_outputs.parquet 中沒有 y_growth，無法計算 metrics")

    y_true = df["y_growth"].values
    y_pred = df["y_growth_pred"].values

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    print(f"[ALL] RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    metrics = {
        "rmse_all": rmse,
        "mae_all": mae,
        "r2_all": r2,
    }
    with open("data/processed/regression_metrics_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[evaluate_regression] Saved regression_metrics_eval.json")


if __name__ == "__main__":
    main()
