#!/usr/bin/env python3

import boto3

def main():
    # Change these as needed
    region = "eu-west-2"
    account_id = "637423166046"
    repository = "my-rag-inference-repo"
    image_tag = "v1"
    role_arn = "arn:aws:iam::637423166046:role/service-role/AmazonSageMaker-ExecutionRole-20241216T172309"

    # Construct the ECR image URI
    ecr_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository}:{image_tag}"
    print("Using ECR image:", ecr_image_uri)

    # We'll name this model "rag-model-minimal-v1"
    model_name = "rag-model-minimal-v1"

    # Environment variables for your container
    # These tell your container where to download the files from S3
    container_env = {
        "FAISS_INDEX_S3": "s3://aym-client-data-in/rag/faiss_index.bin",
        "METADATA_S3":    "s3://aym-client-data-in/rag/index_metadata.json",
        "SUMM_MODEL_S3":  "s3://aym-client-data-in/my_summarization_model.tar.gz",
        # If you want to override the default embedding or generation model name, add:
        # "EMBED_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
        # "GEN_MODEL_NAME": "/opt/ml/model/summarization_model",
    }

    sm_client = boto3.client("sagemaker", region_name=region)

    # Create the SageMaker Model
    response = sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": ecr_image_uri,
            "Environment": container_env,
            "Mode": "SingleModel"
        },
        ExecutionRoleArn=role_arn
    )

    print("Created SageMaker Model:", response["ModelArn"])


if __name__ == "__main__":
    main()

