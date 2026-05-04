"""
Task 4 — Retraining Pipeline
Combines old + new data, retrains champion model type, promotes only if MAE
improves by >= 0.3 on the original test set.
Saves results/step4_s8.json
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV     = os.path.join(BASE_DIR, "data", "training_data.csv")
NEW_CSV       = os.path.join(BASE_DIR, "data", "new_data.csv")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
RESULT_DIR    = os.path.join(BASE_DIR, "results")
STEP1_JSON    = os.path.join(RESULT_DIR, "step1_s1.json")
CHAMPION_PKL  = os.path.join(MODEL_DIR, "champion_model.pkl")

FEATURES          = ["temperature_c", "building_sqm", "occupancy_pct", "is_weekday"]
TARGET            = "energy_kwh"
MIN_IMPROVEMENT   = 0.3   # MAE must improve by at least this much to promote


def get_model_class(name: str):
    return RandomForestRegressor if name == "RandomForest" else GradientBoostingRegressor


def retrain():
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # ── Read original & new data ───────────────────────────────────────────────
    df_orig = pd.read_csv(TRAIN_CSV)
    df_new  = pd.read_csv(NEW_CSV)
    df_comb = pd.concat([df_orig, df_new], ignore_index=True)

    n_orig = len(df_orig)
    n_new  = len(df_new)
    n_comb = len(df_comb)
    print(f"Original rows : {n_orig}")
    print(f"New rows      : {n_new}")
    print(f"Combined rows : {n_comb}")

    # ── Hold-out split on ORIGINAL data (same split as Task 1) ────────────────
    X_orig, y_orig = df_orig[FEATURES], df_orig[TARGET]
    _, X_test, _, y_test = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42
    )

    # ── Champion model (from Task 2) ───────────────────────────────────────────
    champion = joblib.load(CHAMPION_PKL)
    champion_preds = champion.predict(X_test)
    champion_mae   = mean_absolute_error(y_test, champion_preds)
    print(f"\nChampion MAE on original test set: {champion_mae:.4f}")

    # ── Determine model type from Task 1 result ────────────────────────────────
    with open(STEP1_JSON) as f:
        step1 = json.load(f)
    best_name  = step1["best_model"]
    ModelClass = get_model_class(best_name)
    print(f"Retraining model type: {best_name}")

    # ── Retrain on combined data, same train/test philosophy ──────────────────
    X_comb, y_comb = df_comb[FEATURES], df_comb[TARGET]
    X_comb_train, _, y_comb_train, _ = train_test_split(
        X_comb, y_comb, test_size=0.2, random_state=42
    )

    # Use the best hyper-params saved by tune.py if available
    step2_json = os.path.join(RESULT_DIR, "step2_s2.json")
    if os.path.exists(step2_json):
        with open(step2_json) as f:
            step2 = json.load(f)
        best_params = step2["best_params"]
        print(f"Using tuned params: {best_params}")
    else:
        best_params = {"n_estimators": 100, "max_depth": 7, "min_samples_split": 2}
        print("step2_s2.json not found — using default params")

    retrained = ModelClass(random_state=42, **best_params)
    retrained.fit(X_comb_train, y_comb_train)

    # Evaluate retrained model on the SAME original test set
    retrained_preds = retrained.predict(X_test)
    retrained_mae   = mean_absolute_error(y_test, retrained_preds)
    improvement     = champion_mae - retrained_mae   # positive = better
    print(f"Retrained MAE on original test set: {retrained_mae:.4f}")
    print(f"Improvement: {improvement:.4f}  (threshold={MIN_IMPROVEMENT})")

    # ── Promotion decision ────────────────────────────────────────────────────
    if improvement >= MIN_IMPROVEMENT:
        action = "promoted"
        joblib.dump(retrained, CHAMPION_PKL)
        print("✅ Retrained model PROMOTED — champion updated.")
    else:
        action = "kept_champion"
        print("⚠️  Retrained model NOT promoted — champion retained.")

    # ── Save result JSON ───────────────────────────────────────────────────────
    output = {
        "original_data_rows":   n_orig,
        "new_data_rows":        n_new,
        "combined_data_rows":   n_comb,
        "champion_mae":         round(champion_mae,   4),
        "retrained_mae":        round(retrained_mae,  4),
        "improvement":          round(improvement,    4),
        "min_improvement_threshold": MIN_IMPROVEMENT,
        "action":               action,
        "comparison_metric":    "mae",
    }
    out_path = os.path.join(RESULT_DIR, "step4_s8.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    retrain()