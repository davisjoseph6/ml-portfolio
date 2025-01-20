FROM 763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04-v1.3

# Switch to bash so we can source conda env
SHELL ["/bin/bash", "-c"]

# 1) Install pinned Transformers, Sentence Transformers, plus Faiss
RUN source activate base && \
    pip install --no-cache-dir \
        "transformers==4.28.1" \
        "sentence-transformers<2.2.0" \
        "faiss-cpu==1.7.3"

# 2) Now force-upgrade huggingface-hub so snapshot_download is available
RUN source activate base && \
    pip install --no-cache-dir --upgrade --force-reinstall "huggingface-hub==0.15.1"

# 3) Verify huggingface-hub version
RUN source activate base && \
    pip show huggingface-hub && \
    python -c "import huggingface_hub; print('huggingface_hub version:', huggingface_hub.__version__)"

# 4) Copy your custom inference script
COPY inference.py /opt/ml/model/code/inference.py

# 5) Environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV FAISS_INDEX_S3="s3://aym-client-data-in/rag/faiss_index.bin"
ENV METADATA_S3="s3://aym-client-data-in/rag/index_metadata.json"
ENV EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
ENV GEN_MODEL_NAME="my_summarization_model"

