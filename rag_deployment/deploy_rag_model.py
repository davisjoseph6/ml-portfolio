#!/usr/bin/env python3

import boto3

def main():
    # 1) Specify region, account, ECR repo, and image tag
    region = "eu-west-2"
    account_id = "637423166046"
    ecr_repository = "my-rag-inference-repo"
    image_tag = "v4"  # Updated tag

    # 2) Construct the ECR image URI
    ecr_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{ecr_repository}:{image_tag}"
    print("Using ECR image:", ecr_image_uri)

    # 3) SageMaker Execution Role (must have permission to pull image & read S3)
    role_arn = "arn:aws:iam::637423166046:role/service-role/AmazonSageMaker-ExecutionRole-20241216T172309"

    # 4) Create SageMaker client
    sm_client = boto3.client("sagemaker", region_name=region)

    # 5) Name your SageMaker Model (e.g. rag-model-v4)
    model_name = "rag-model-v4"

    # 6) Container environment variables
    container_env = {
        "FAISS_INDEX_S3":    "s3://aym-client-data-in/rag/faiss_index.bin",
        "METADATA_S3":       "s3://aym-client-data-in/rag/index_metadata.json",
        "EMBED_MODEL_NAME":  "sentence-transformers/all-MiniLM-L6-v2",
        "GEN_MODEL_NAME":    "my_summarization_model",
        # Ensure SageMaker Inference Toolkit sees your script
        "SAGEMAKER_PROGRAM": "inference.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code"
    }

    # 7) Create (or update) the SageMaker Model
    response = sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": ecr_image_uri,
            "Environment": container_env
        },
        ExecutionRoleArn=role_arn
    )

    print("Created SageMaker Model:", response["ModelArn"])


if __name__ == "__main__":
    main()

