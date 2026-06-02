# EcoSort — inference web app (CPU-only) for Railway deployment.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# libgomp1 is required by torch / scikit-learn OpenMP runtimes.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (much smaller than the default CUDA build),
# then the remaining requirements (torch/torchvision are already satisfied).
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# Single worker (model held in memory once); threads handle concurrent requests.
CMD ["sh", "-c", "gunicorn main:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180"]
