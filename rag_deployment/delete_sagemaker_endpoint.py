#!/usr/bin/env python3

import boto3

def main():
    region = "eu-west-2"
    endpoint_name = "rag-endpoint-v3"
    endpoint_config_name = "rag-endpoint-config-v3"

    sm_client = boto3.client("sagemaker", region_name=region)

    # Delete the endpoint
    print(f"Deleting endpoint: {endpoint_name} if it exists...")
    try:
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint {endpoint_name} deleted.")
    except sm_client.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e):
            print(f"Endpoint {endpoint_name} not found, nothing to delete.")
        else:
            raise

    # Delete endpoint config
    print(f"Deleting endpoint config: {endpoint_config_name} if it exists...")
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"Endpoint config {endpoint_config_name} deleted.")
    except sm_client.exceptions.ClientError as e:
        if "Could not find endpoint configuration" in str(e):
            print(f"Endpoint config {endpoint_config_name} not found, nothing to delete.")
        else:
            raise

if __name__ == "__main__":
    main()

