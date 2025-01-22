FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ git libblas-dev liblapack-dev ca-certificates tar \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    torch==2.0.1 \
    faiss-cpu==1.7.3 \
    flask \
    gunicorn \
    boto3 \
    git+https://github.com/UKPLab/sentence-transformers.git@master

RUN mkdir -p /opt/ml/model

COPY app.py /opt/ml/code/app.py
COPY serve.sh /usr/local/bin/serve.sh
RUN chmod +x /usr/local/bin/serve.sh

WORKDIR /opt/ml/code

ENTRYPOINT ["/usr/local/bin/serve.sh"]

EXPOSE 8080

