"""
Task 2 — Hyperparameter Tuning
Grid-search the best model from Task 1 with nested MLflow runs.
Saves results/step2_s2.json
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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "training_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
STEP1_JSON = os.path.join(RESULT_DIR, "step1_s1.json")

FEATURES = ["temperature_c", "building_sqm", "occupancy_pct", "is_weekday"]
TARGET   = "energy_kwh"
EXP_NAME = "powergrid-energy-kwh"

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 7, 15],
    "min_samples_split": [2, 4],
}
N_FOLDS = 3
PARENT_NAME = "tuning-powergrid"


def get_model_class(name: str):
    return RandomForestRegressor if name == "RandomForest" else GradientBoostingRegressor


def make_all_param_combos(grid: dict) -> list:
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def tune():
    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES].values, df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ── Determine winner from Task 1 ──────────────────────────────────────────
    with open(STEP1_JSON) as f:
        step1 = json.load(f)

    best_name = step1["best_model"]
    ModelClass = get_model_class(best_name)

    print(f"Tuning model: {best_name}")

    # ── MLflow setup (Windows-safe fix) ───────────────────────────────────────
    mlruns_dir = Path(BASE_DIR, "mlruns")
    mlruns_dir.mkdir(exist_ok=True)

    tracking_uri = mlruns_dir.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXP_NAME)

    combos = make_all_param_combos(PARAM_GRID)
    total_trials = len(combos)
    kf = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=42
    )

    best_rmse = float("inf")
    best_cv_mae = float("inf")
    best_mae = float("inf")
    best_params = {}

    # ── Parent run ─────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=PARENT_NAME):
        mlflow.set_tag("tuning", "grid_search")
        mlflow.log_param("model_type", best_name)
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("total_trials", total_trials)

        for i, params in enumerate(combos, 1):
            with mlflow.start_run(
                run_name=f"trial_{i}",
                nested=True
            ):
                model = ModelClass(
                    random_state=42,
                    **params
                )

                # CV
                cv_neg_mae = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=kf,
                    scoring="neg_mean_absolute_error"
                )

                cv_mae = -cv_neg_mae.mean()

                # Train + test
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                mae = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))

                # Log
                for k, v in params.items():
                    mlflow.log_param(k, v)

                mlflow.log_metric("cv_mae", round(cv_mae, 4))
                mlflow.log_metric("test_mae", round(mae, 4))
                mlflow.log_metric("test_rmse", round(rmse, 4))

                print(
                    f"Trial {i}/{total_trials} | "
                    f"{params} | "
                    f"cv_mae={cv_mae:.3f} | "
                    f"rmse={rmse:.3f}"
                )

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_cv_mae = cv_mae
                    best_mae = mae
                    best_params = params

        # parent summary
        mlflow.log_params({
            f"best_{k}": v for k, v in best_params.items()
        })
        mlflow.log_metric("best_test_rmse", round(best_rmse, 4))
        mlflow.log_metric("best_test_mae", round(best_mae, 4))

    print("\nBest params:", best_params)
    print("Best RMSE:", round(best_rmse, 4))
    print("Best MAE:", round(best_mae, 4))

    # Save tuned model
    best_model = ModelClass(
        random_state=42,
        **best_params
    )

    best_model.fit(X_train, y_train)

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "tuned_model.pkl")
    )

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "champion_model.pkl")
    )

    # Save JSON
    output = {
        "search_type": "grid",
        "n_folds": N_FOLDS,
        "total_trials": total_trials,
        "best_params": best_params,
        "best_mae": round(best_mae, 4),
        "best_cv_mae": round(best_cv_mae, 4),
        "parent_run_name": PARENT_NAME,
    }

    out_path = os.path.join(
        RESULT_DIR,
        "step2_s2.json"
    )

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved -> {out_path}")


if __name__ == "__main__":
    tune()