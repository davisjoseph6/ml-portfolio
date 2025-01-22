#!/usr/bin/env python3

import boto3

def main():
    region = "eu-west-2"
    sm_client = boto3.client("sagemaker", region_name=region)

    # Match the ModelName you created in deploy_rag_model.py
    model_name = "rag-model-minimal-v1"

    # Create new EndpointConfig and Endpoint
    endpoint_config_name = "rag-endpoint-config-minimal-v1"
    endpoint_name        = "rag-endpoint-minimal-v1"

    # 1) Create endpoint config
    print(f"Creating endpoint config: {endpoint_config_name}")
    try:
        create_config_resp = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge",  # or ml.c5.xlarge if you prefer
                    "InitialVariantWeight": 1.0
                }
            ]
        )
        print("Endpoint config ARN:", create_config_resp["EndpointConfigArn"])
    except sm_client.exceptions.ClientError as e:
        if "EndpointConfigNameAlreadyExists" in str(e):
            print("Endpoint config already exists. Reusing. Or delete first.")
        else:
            raise e

    # 2) Create endpoint
    print(f"Creating endpoint: {endpoint_name}")
    try:
        create_endpoint_resp = sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print("Endpoint ARN:", create_endpoint_resp["EndpointArn"])
    except sm_client.exceptions.ClientError as e:
        if "EndpointAlreadyExists" in str(e):
            print("Endpoint already exists. Reusing it. Or delete first.")
        else:
            raise e

    # 3) Wait for endpoint to be InService
    print(f"Waiting for endpoint {endpoint_name} to become InService...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    print(f"Endpoint {endpoint_name} is now InService!")


if __name__ == "__main__":
    main()

