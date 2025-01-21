#!/usr/bin/env python3

import boto3

def main():
    region = "eu-west-2"
    account_id = "637423166046"
    ecr_repository = "my-rag-inference-repo"
    image_tag = "v-minimal"

    ecr_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{ecr_repository}:{image_tag}"
    print("Using ECR image:", ecr_image_uri)

    role_arn = "arn:aws:iam::637423166046:role/service-role/AmazonSageMaker-ExecutionRole-20241216T172309"
    sm_client = boto3.client("sagemaker", region_name=region)

    model_name = "rag-model-minimal"

    container_env = {
        "FAISS_INDEX_S3": "s3://aym-client-data-in/rag/faiss_index.bin",
        "METADATA_S3":    "s3://aym-client-data-in/rag/index_metadata.json",
        "EMBED_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
        "GEN_MODEL_NAME": "my_summarization_model",
        # Tells sagemaker_inference to use /opt/ml/model/code/inference.py
        "SAGEMAKER_PROGRAM": "inference.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code"
    }

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

