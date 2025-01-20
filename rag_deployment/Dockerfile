# Dockerfile

FROM 763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04-v1.3

# 1) Install pinned Transformers & Sentence Transformers, plus Faiss
RUN pip install --no-cache-dir \
    "transformers==4.28.1" \
    "sentence-transformers<2.2.0" \
    faiss-cpu==1.7.3

# 2) Now force-upgrade huggingface-hub so snapshot_download is available
RUN pip install --no-cache-dir \
    huggingface-hub==0.15.1

# 3) Copy your custom inference script into the container
COPY inference.py /opt/ml/model/code/inference.py

# 4) Environment variables (so your container code knows where to load from)
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV FAISS_INDEX_S3="s3://aym-client-data-in/rag/faiss_index.bin"
ENV METADATA_S3="s3://aym-client-data-in/rag/index_metadata.json"
ENV EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
ENV GEN_MODEL_NAME="my_summarization_model"

