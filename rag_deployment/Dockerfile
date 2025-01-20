# Dockerfile

FROM public.ecr.aws/huggingface/transformers-pytorch-inference:2.1.0-pytorch-cpu-py39

# Additional packages
RUN pip install --no-cache-dir faiss-cpu==1.7.3 sentence-transformers

# Copy your custom inference code
COPY inference.py /opt/ml/model/code/inference.py

# Environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV FAISS_INDEX_S3="s3://aym-client-data-in/rag/faiss_index.bin"
ENV METADATA_S3="s3://aym-client-data-in/rag/index_metadata.json"
ENV EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
ENV GEN_MODEL_NAME="my_summarization_model"

