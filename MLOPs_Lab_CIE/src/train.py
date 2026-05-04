"""
Task 1 — Experiment Tracking & Model Comparison
Trains RandomForest and GradientBoosting, logs to MLflow, saves results/step1_s1.json
"""

import os
import json
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

FEATURES = ["temperature_c", "building_sqm", "occupancy_pct", "is_weekday"]
TARGET = "energy_kwh"
EXP_NAME = "powergrid-energy-kwh"


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(
        np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))
    ) * 100

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4),
    }


def train():
    # Load data
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # MLflow setup (correct Windows URI)
    mlruns_dir = Path(BASE_DIR, "mlruns")
    mlruns_dir.mkdir(exist_ok=True)

    tracking_uri = mlruns_dir.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXP_NAME)

    # Models
    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=7,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            random_state=42
        ),
    }

    results = []

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("project_phase", "model_selection")

            for k, v in model.get_params().items():
                mlflow.log_param(k, v)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = compute_metrics(y_test.values, preds)

            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("r2", metrics["r2"])
            mlflow.log_metric("mape", metrics["mape"])

            mlflow.sklearn.log_model(model, artifact_path=name)

            results.append({
                "name": name,
                **metrics
            })

            print(
                f"[{name}] "
                f"MAE={metrics['mae']} "
                f"RMSE={metrics['rmse']} "
                f"R2={metrics['r2']} "
                f"MAPE={metrics['mape']}"
            )

    # Pick best
    best = min(results, key=lambda x: x["rmse"])
    best_model_name = best["name"]

    best_estimator = candidates[best_model_name]
    best_estimator.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, "champion_model.pkl")
    joblib.dump(best_estimator, model_path)

    print(f"\nBest model: {best_model_name}")
    print(f"Saved champion model -> {model_path}")

    output = {
        "experiment_name": EXP_NAME,
        "models": results,
        "best_model": best_model_name,
        "best_metric_name": "rmse",
        "best_metric_value": best["rmse"],
    }

    out_path = os.path.join(RESULT_DIR, "step1_s1.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved -> {out_path}")


if __name__ == "__main__":
    train()