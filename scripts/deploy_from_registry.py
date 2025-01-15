#!/usr/bin/env python3

import boto3

def main():
    region = "eu-west-2"
    model_package_arn = "arn:aws:sagemaker:eu-west-2:637423166046:model-package/SummarizationModelsGroup/1"
    role_arn = "arn:aws:iam::637423166046:role/service-role/AmazonSageMaker-ExecutionRole-20241216T172309"

    sm_client = boto3.client("sagemaker", region_name=region)

    # 1. Create Model
    model_name = "my-summarization-registry-model"
    print(f"Creating SageMaker Model: {model_name}")

    create_model_response = sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "ModelPackageName": model_package_arn
        },
        ExecutionRoleArn=role_arn
    )
    print("Model created:", create_model_response["ModelArn"])

    region = "eu-west-2"
    sm_client = boto3.client("sagemaker", region_name=region)

    endpoint_config_name = "my-summar-endpoint-config"
    model_name = "my-summarization-registry-model"  # from Step 2

    print(f"Creating Endpoint Config: {endpoint_config_name}")

    response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.m5.large",
                "InitialVariantWeight": 1.0
            }
        ]
    )
    print("Endpoint Config created:", response["EndpointConfigArn"])

if __name__ == "__main__":
    main()

