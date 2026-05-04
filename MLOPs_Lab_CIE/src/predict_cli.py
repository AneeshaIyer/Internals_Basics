"""
Task 3 — Docker Packaging
CLI predictor: accepts features via argparse, loads champion model, prints JSON prediction.

Usage (inside container or locally):
    python predict_cli.py --temperature_c 33.9 --building_sqm 327.1 \
                          --occupancy_pct 45.9 --is_weekday 1
"""

import argparse
import json
import os
import sys

import numpy as np
import joblib

# ── Paths (works both inside Docker /app and local project root) ──────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR   = os.path.dirname(_SCRIPT_DIR)          # one level up from src/
MODEL_PATH  = os.path.join(_BASE_DIR, "models", "champion_model.pkl")
RESULT_DIR  = os.path.join(_BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

FEATURES = ["temperature_c", "building_sqm", "occupancy_pct", "is_weekday"]


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def predict(args):
    model = load_model()

    features = np.array([[
        args.temperature_c,
        args.building_sqm,
        args.occupancy_pct,
        args.is_weekday,
    ]])

    prediction = float(model.predict(features)[0])

    # ── Build & print result ──────────────────────────────────────────────────
    result = {
        "image_name": "powergrid-predictor",
        "image_tag":  "v1",
        "base_image": "python:3.10-slim",
        "test_input": {
            "temperature_c": args.temperature_c,
            "building_sqm":  args.building_sqm,
            "occupancy_pct": args.occupancy_pct,
            "is_weekday":    args.is_weekday,
        },
        "prediction": round(prediction, 4),
    }

    print(json.dumps(result, indent=2))

    # Persist to results/step3_s3.json (only meaningful outside Docker)
    out_path = os.path.join(RESULT_DIR, "step3_s3.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="PowerGrid Energy Consumption Predictor"
    )
    parser.add_argument("--temperature_c",  type=float, required=True,
                        help="Outdoor temperature in Celsius (15–45)")
    parser.add_argument("--building_sqm",   type=float, required=True,
                        help="Building area in square metres (50–500)")
    parser.add_argument("--occupancy_pct",  type=float, required=True,
                        help="Occupancy percentage (20–100)")
    parser.add_argument("--is_weekday",     type=int,   required=True,
                        choices=[0, 1],
                        help="1 if weekday, 0 if weekend")
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())