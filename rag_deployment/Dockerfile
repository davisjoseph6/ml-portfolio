FROM 763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04-v1.3

SHELL ["/bin/bash", "-c"]

# 1) Copy your code
COPY inference.py /opt/ml/model/code/inference.py
COPY requirements.txt /opt/ml/model/code/requirements.txt

# 2) Install packages in /opt/conda
RUN /opt/conda/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/conda/bin/pip install --no-cache-dir -r /opt/ml/model/code/requirements.txt

# 3) Verify huggingface_hub has snapshot_download
RUN /opt/conda/bin/pip freeze | grep huggingface-hub
RUN /opt/conda/bin/python -c "from huggingface_hub import snapshot_download; print('snapshot_download is available!')"

# 4) Set environment variables
ENV SAGEMAKER_PROGRAM=inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code

# 5) Also set your FAISS_INDEX_S3, METADATA_S3, etc., if you wish
ENV FAISS_INDEX_S3="s3://aym-client-data-in/rag/faiss_index.bin"
ENV METADATA_S3="s3://aym-client-data-in/rag/index_metadata.json"
ENV EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
ENV GEN_MODEL_NAME="my_summarization_model"

