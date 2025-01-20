# Dockerfile
FROM 763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04-v1.3

# Install additional packages needed:
#  - Pin Transformers to 4.28.1 (same as container)
#  - Pin sentence-transformers below 2.2.0 to avoid pulling later Transformers versions
#  - Install Faiss CPU
#  - Explicitly install huggingface-hub so we have snapshot_download available
RUN pip install --no-cache-dir \
    "transformers==4.28.1" \
    "sentence-transformers<2.2.0" \
    faiss-cpu==1.7.3 \
    huggingface-hub==0.15.1

# Copy your custom inference script into the container
COPY inference.py /opt/ml/model/code/inference.py

# Environment variables for clarity
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV FAISS_INDEX_S3="s3://aym-client-data-in/rag/faiss_index.bin"
ENV METADATA_S3="s3://aym-client-data-in/rag/index_metadata.json"
ENV EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
ENV GEN_MODEL_NAME="my_summarization_model"
