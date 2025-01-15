#!/usr/bin/env python3

import boto3
import json
import time

def main():
    region = "eu-west-2"
    sm_client = boto3.client("sagemaker", region_name=region)

    endpoint_name = "my-summar-endpoint"             # give your endpoint a name
    endpoint_config_name = "my-summar-endpoint-config"  # same name from Step 3

    print(f"Creating Endpoint: {endpoint_name}")

    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    print("Endpoint creation initiated.")
    print("Endpoint ARN:", create_endpoint_response["EndpointArn"])

    # Optionally wait for the endpoint to become InService
    print("Waiting for endpoint to be InService... (this can take a few minutes)")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    print("Endpoint is now InService!")

if __name__ == "__main__":
    main()

