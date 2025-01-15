#!/usr/bin/env python3

import boto3
import time
from botocore.exceptions import ClientError

def main():
    sm_client = boto3.client("sagemaker", region_name="eu-west-2")
    package_group_name = "SummarizationModelsGroup"

    # 1. Check or create the Model Package Group
    try:
        sm_client.describe_model_package_group(ModelPackageGroupName=package_group_name)
        print(f"Model Package Group '{package_group_name}' already exists. Proceeding...")
    except ClientError as e:
        if "ResourceNotFound" in str(e):
            sm_client.create_model_package_group(
                ModelPackageGroupName=package_group_name,
                ModelPackageGroupDescription="Models for text summarization."
            )
            print(f"Created Model Package Group: {package_group_name}")
        else:
            raise e

    # 2. Use a valid container tag for eu-west-2
    inference_image_uri = (
        "763104351884.dkr.ecr.eu-west-2.amazonaws.com/"
        "huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04-v1.6"
    )
    model_data_url = "s3://aym-client-data-in/my_summarization_model.tar.gz"

    # 3. Create a new model package
    timestamp = int(time.time())
    model_package_name = f"{package_group_name}-v1-{timestamp}"

    response = sm_client.create_model_package(
        ModelPackageGroupName=package_group_name,
        ModelPackageDescription="Version 1 of Summarization Model.",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": inference_image_uri,
                    "ModelDataUrl": model_data_url,
                    "Environment": {
                        "HF_TASK": "summarization"
                    }
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"]
        },
        ModelApprovalStatus="PendingManualApproval",
    )

    arn = response["ModelPackageArn"]
    print(f"Created Model Package: {arn}")

    # 4. Approve it immediately (optional)
    sm_client.update_model_package(
        ModelPackageArn=arn,
        ModelApprovalStatus="Approved"
    )
    print("Model Package Approved!")

if __name__ == "__main__":
    main()

