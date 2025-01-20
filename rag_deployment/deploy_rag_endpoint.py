#!/usr/bin/env python3

import boto3
import time

def main():
    region = "eu-west-2"

    # Must match your new model name from deploy_rag_model.py
    model_name = "rag-model-v4"

    # We'll create an endpoint config and endpoint for "v4"
    endpoint_config_name = "rag-endpoint-config-v4"
    endpoint_name = "rag-endpoint-v4"

    # Create a SageMaker client
    sm_client = boto3.client("sagemaker", region_name=region)

    # 1) Create Endpoint Config
    print(f"Creating endpoint config: {endpoint_config_name}")
    try:
        create_config_resp = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge",  # or ml.g4dn.xlarge for GPU
                    "InitialVariantWeight": 1.0
                }
            ]
        )
        print("Endpoint config ARN:", create_config_resp["EndpointConfigArn"])
    except sm_client.exceptions.ClientError as e:
        if "EndpointConfigNameAlreadyExists" in str(e):
            print("Endpoint config already exists. Reusing or consider deleting first.")
        else:
            raise e

    # 2) Create Endpoint
    print(f"Creating endpoint: {endpoint_name}")
    try:
        create_endpoint_resp = sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print("Endpoint ARN:", create_endpoint_resp["EndpointArn"])
    except sm_client.exceptions.ClientError as e:
        if "EndpointAlreadyExists" in str(e):
            print("Endpoint already exists. Reusing it or consider deleting first.")
        else:
            raise e

    # 3) Wait for Endpoint to be InService
    print(f"Waiting for endpoint {endpoint_name} to become InService...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    print(f"Endpoint {endpoint_name} is now InService!")

if __name__ == "__main__":
    main()

