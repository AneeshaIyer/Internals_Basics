# Internals_Basics
MLOps Lab CIE Submission
# ⚡ PowerGrid Energy Prediction — MLOps Pipeline

**Course:** MLOps (24AM6AEMLO)  
**College:** BMS College of Engineering  
**Semester:** VI - 2026 Even  
**Name:** Aneesha Manjunath Iyer
**USN:** 1BM23AI021

---

## 🧠 Project Overview

This project builds an end-to-end MLOps pipeline to predict energy consumption (kWh) for commercial buildings managed by PowerGrid.

The goal is to support load balancing by accurately forecasting electricity demand using building and environmental features.

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `temperature_c` | Outdoor temperature (15–45) |
| `building_sqm` | Building area in square meters (50–500) |
| `occupancy_pct` | Occupancy percentage (20–100) |
| `is_weekday` | Whether it is a weekday (0 or 1) |
| `energy_kwh` | Target — energy consumption |

---

## 🚀 Tasks

### Task 1 — Experiment Tracking & Model Comparison
- Trained RandomForest and GradientBoosting models
- Logged MAE, RMSE, R², MAPE using MLflow
- Best model selected based on RMSE

---

### Task 2 — Hyperparameter Tuning
- Grid search over RandomForest parameters:
  - n_estimators: [100, 200, 300]
  - max_depth: [3, 7, 15]
  - min_samples_split: [2, 4]
- 3-fold cross-validation (18 trials)
- Logged as nested MLflow runs under `tuning-powergrid`

---

### Task 3 — Docker Packaging
- Built Docker image: `powergrid-predictor:v1`
- Created CLI-based prediction system using argparse
- Model loaded inside container for inference

---

### Task 4 — Retraining Pipeline
- Combined original dataset + new incoming data
- Retrained champion model type from Task 1
- Compared performance on same test set
- Model promoted only if MAE improves by ≥ 0.3

---

## 📁 Results

| Task | Output File |
|---|---|
| Task 1 | `results/step1_s1.json` |
| Task 2 | `results/step2_s2.json` |
| Task 3 | `results/step3_s3.json` |
| Task 4 | `results/step4_s8.json` |

---

## ⚙️ How to Run

```bash
cd MLOps_Lab_CIE
python src/train.py
python src/tune.py
docker build -t powergrid-predictor:v1 .
docker run powergrid-predictor:v1 --temperature_c 33.9 --building_sqm 327.1 --occupancy_pct 45.9 --is_weekday 1
python src/retrain.py
