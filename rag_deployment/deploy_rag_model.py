#!/usr/bin/env python3

import boto3

def main():
    region = "eu-west-2"
    account_id = "637423166046"
    ecr_repository = "my-rag-inference-repo"
    image_tag = "v1"

    # The ECR image URI we pushed
    ecr_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{ecr_repository}:{image_tag}"
    print("Using ECR image:", ecr_image_uri)

    # SageMaker Execution Role that allows:
    #  1) Download from S3
    #  2) Write CloudWatch logs
    #  3) Possibly pull images from ECR
    role_arn = "arn:aws:iam::637423166046:role/service-role/AmazonSageMaker-ExecutionRole-20241216T172309"

    # Create a SageMaker client
    sm_client = boto3.client("sagemaker", region_name=region)

    # Name your SageMaker Model
    model_name = "rag-model-v1"

    # (Optional) You can override environment variables for the container:
    container_env = {
        "FAISS_INDEX_S3": "s3://aym-client-data-in/rag/faiss_index.bin",
        "METADATA_S3":    "s3://aym-client-data-in/rag/index_metadata.json",
        "EMBED_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
        "GEN_MODEL_NAME":   "my_summarization_model"
    }

    # 1) Create SageMaker Model
    response = sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": ecr_image_uri,
            "Environment": container_env
        },
        ExecutionRoleArn=role_arn
    )
    print("Created/Updated SageMaker Model:", response["ModelArn"])

if __name__ == "__main__":
    main()

