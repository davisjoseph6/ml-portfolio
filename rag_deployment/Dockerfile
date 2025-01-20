# Dockerfile

FROM 763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04-v1.3

# Copy your custom inference script
COPY inference.py /opt/ml/model/code/inference.py

# Copy your requirements.txt to the same folder
COPY requirements.txt /opt/ml/model/code/requirements.txt

# Tell SageMaker which script is our entry script, and to install from requirements
ENV SAGEMAKER_PROGRAM="inference.py"
ENV SAGEMAKER_REQUIREMENTS="requirements.txt"

# (Optional) Additional environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# For your RAG environment config
ENV FAISS_INDEX_S3="s3://aym-client-data-in/rag/faiss_index.bin"
ENV METADATA_S3="s3://aym-client-data-in/rag/index_metadata.json"
ENV EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
ENV GEN_MODEL_NAME="my_summarization_model"

