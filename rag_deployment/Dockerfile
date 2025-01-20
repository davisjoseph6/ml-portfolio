# Dockerfile
FROM 763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04-v1.3

# 1) Install your core libraries
#    - Transformers pinned to 4.28.1 (matches base image)
#    - sentence-transformers pinned below 2.2.0 (if needed)
#    - Faiss CPU
RUN pip install --no-cache-dir \
    "transformers==4.28.1" \
    "sentence-transformers<2.2.0" \
    faiss-cpu==1.7.3

# 2) Force upgrade huggingface-hub so we have snapshot_download
#    (0.27.1 is just an example version known to support snapshot_download)
RUN pip install --no-cache-dir --upgrade huggingface-hub==0.27.1

# 3) Print out the final environment for debugging in build logs
RUN pip freeze

# 4) Copy your custom inference script into the container
COPY inference.py /opt/ml/model/code/inference.py

# 5) Environment variables for clarity and default config
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV FAISS_INDEX_S3="s3://aym-client-data-in/rag/faiss_index.bin"
ENV METADATA_S3="s3://aym-client-data-in/rag/index_metadata.json"
ENV EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
ENV GEN_MODEL_NAME="my_summarization_model"

