# Dockerfile

FROM 763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04-v1.3

# Use bash so we can run conda commands if we want
SHELL ["/bin/bash", "-c"]

# 1) (Optional) Show which conda envs exist (for debugging)
RUN conda env list

# 2) Install pinned Transformers, Sentence Transformers, plus Faiss
#    using the conda python environment explicitly
RUN /opt/conda/bin/pip install --no-cache-dir \
    "transformers==4.28.1" \
    "sentence-transformers<2.2.0" \
    "faiss-cpu==1.7.3"

# 3) Now force-upgrade huggingface-hub so snapshot_download is available
RUN /opt/conda/bin/pip install --no-cache-dir --upgrade --force-reinstall "huggingface-hub==0.15.1"

# 4) Verify huggingface-hub version
RUN /opt/conda/bin/pip show huggingface-hub
RUN /opt/conda/bin/python -c "import huggingface_hub; print('huggingface_hub version:', huggingface_hub.__version__)"

# 5) Copy your custom inference script
COPY inference.py /opt/ml/model/code/inference.py

# 6) (Optional) Ensure /opt/conda/bin is in PATH at runtime
ENV PATH="/opt/conda/bin:$PATH"

# 7) Environment variables (so your container code knows where to load from)
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV FAISS_INDEX_S3="s3://aym-client-data-in/rag/faiss_index.bin"
ENV METADATA_S3="s3://aym-client-data-in/rag/index_metadata.json"
ENV EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
ENV GEN_MODEL_NAME="my_summarization_model"

