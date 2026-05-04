# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Working directory inside container ────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (layer-cache friendly) ────────────────────────────
COPY requirements.txt .

# ── Install dependencies ───────────────────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY src/     ./src/
COPY models/  ./models/
COPY results/ ./results/

# ── Default entry-point: run the CLI predictor ────────────────────────────────
ENTRYPOINT ["python", "src/predict_cli.py"]