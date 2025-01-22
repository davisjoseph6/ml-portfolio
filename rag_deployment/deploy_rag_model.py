#!/usr/bin/env python3

import boto3

def main():
    region = "eu-west-2"
    account_id = "637423166046"
    repository = "my-rag-inference-repo"
    image_tag = "v1"

    role_arn = "arn:aws:iam::637423166046:role/service-role/AmazonSageMaker-ExecutionRole-20241216T172309"
    model_name = "rag-model-minimal-v1"

    ecr_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository}:{image_tag}"
    print("Using ECR image:", ecr_image_uri)

    container_env = {
        "FAISS_INDEX_S3": "s3://aym-client-data-in/rag/faiss_index.bin",
        "METADATA_S3":    "s3://aym-client-data-in/rag/index_metadata.json",
        "SUMM_MODEL_S3":  "s3://aym-client-data-in/my_summarization_model.tar.gz",
        # optional: "EMBED_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
        # optional: "GEN_MODEL_NAME": "/opt/ml/model/summarization_model",
    }

    sm_client = boto3.client("sagemaker", region_name=region)

    response = sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": ecr_image_uri,
            "Environment": container_env,
            "Mode": "SingleModel"
            # No Command/Args here => SageMaker won't override your Dockerfile
        },
        ExecutionRoleArn=role_arn
    )
    print("Created SageMaker Model:", response["ModelArn"])


if __name__ == "__main__":
    main()

