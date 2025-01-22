# Dockerfile
FROM python:3.9-slim

# 1) Install system dependencies (including git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    git \
    libblas-dev \
    liblapack-dev \
    ca-certificates \
    tar \
    && rm -rf /var/lib/apt/lists/*

# 2) Upgrade pip & install Python libraries
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3) Install the main libraries (no pinned transformers)
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    torch==2.0.1 \
    faiss-cpu==1.7.3 \
    flask \
    gunicorn \
    boto3 \
    git+https://github.com/UKPLab/sentence-transformers.git@master

# *** Create /opt/ml/model so Boto3 can write Faiss + metadata here ***
RUN mkdir -p /opt/ml/model

# 4) Copy your Flask app
COPY app.py /opt/ml/code/app.py

WORKDIR /opt/ml/code
EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app", "--timeout", "120"]

