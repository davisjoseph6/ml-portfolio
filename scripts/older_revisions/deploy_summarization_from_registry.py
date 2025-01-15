#!/usr/bin/env python3

import argparse
import time
import boto3
from botocore.exceptions import ClientError


def wait_for_endpoint_in_service(sm_client, endpoint_name):
    """
    Polls the endpoint's status until it is InService or fails.
    Raises an exception if the endpoint goes into a failure or rollback state.
    """
    print(f"Waiting for endpoint '{endpoint_name}' to be InService...")
    while True:
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print(f"  Current status: {status}")
        if status == "InService":
            print(f"Endpoint '{endpoint_name}' is now InService!")
            break
        elif status in ("Failed", "Deleting", "UpdateRollbackFailed"):
            raise RuntimeError(f"Endpoint failed with status: {status}")
        time.sleep(15)  # Wait 15 seconds before polling again


def main():
    parser = argparse.ArgumentParser(
        description="Deploy a Summarization Model from the SageMaker Model Registry."
    )
    parser.add_argument(
        "--model-package-arn",
        type=str,
        required=False,
        default="arn:aws:sagemaker:eu-west-2:123456789012:model-package/SummarizationModelsGroup/1",
        help="The Model Package ARN from the registry (must be in 'Approved' status)."
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default="my-summ-registry-endpoint",
        help="The name of the SageMaker endpoint to create or update."
    )
    parser.add_argument(
        "--region",
        type=str,
        default="eu-west-2",
        help="AWS region (e.g., 'us-east-1')."
    )
    parser.add_argument(
        "--role-arn",
        type=str,
        default="arn:aws:iam::123456789012:role/service-role/MySageMakerExecutionRole",
        help="The IAM role ARN that SageMaker will assume to create and run the model."
    )
    parser.add_argument(
        "--test-invoke",
        action="store_true",
        help="If set, perform a quick test invocation with a sample summarization request."
    )
    args = parser.parse_args()

    # 1. Initialize SageMaker client
    sm_client = boto3.client("sagemaker", region_name=args.region)

    # 2. Create a SageMaker Model that references the Model Package
    model_name = f"summ-registry-model-{int(time.time())}"
    print(f"Creating a SageMaker Model from package: {args.model_package_arn}")
    try:
        create_model_resp = sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "ModelPackageName": args.model_package_arn
            },
            ExecutionRoleArn=args.role_arn
        )
        print("Model created:", create_model_resp["ModelArn"])
    except ClientError as e:
        print(f"Error creating model: {e}")
        return

    # 3. Create an Endpoint Configuration
    endpoint_config_name = f"{args.endpoint_name}-config-{int(time.time())}"
    print(f"Creating endpoint config: {endpoint_config_name}")
    try:
        create_endpoint_config_resp = sm_client.create_endpoint_config(
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
        print("Endpoint config created:", create_endpoint_config_resp["EndpointConfigArn"])
    except ClientError as e:
        print(f"Error creating endpoint config: {e}")
        return

    # 4. Create or Update the Endpoint
    #    If the endpoint doesn't exist, create it. Otherwise, update it.
    print(f"Checking if endpoint '{args.endpoint_name}' exists...")
    endpoint_exists = False
    try:
        sm_client.describe_endpoint(EndpointName=args.endpoint_name)
        endpoint_exists = True
        print(f"Endpoint '{args.endpoint_name}' exists. We will update it.")
    except ClientError as e:
        if "ResourceNotFound" in str(e):
            print(f"Endpoint '{args.endpoint_name}' does not exist. Creating a new one.")
        else:
            print(f"Error describing endpoint: {e}")
            return

    if endpoint_exists:
        print(f"Updating endpoint '{args.endpoint_name}' to use new config: {endpoint_config_name}")
        update_endpoint_resp = sm_client.update_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print("Endpoint update initiated. Response:", update_endpoint_resp)
    else:
        print(f"Creating endpoint '{args.endpoint_name}' with config: {endpoint_config_name}")
        create_endpoint_resp = sm_client.create_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print("Endpoint creation initiated. Response:", create_endpoint_resp)

    # 5. Wait for the endpoint to be InService
    wait_for_endpoint_in_service(sm_client, args.endpoint_name)

    # 6. (Optional) Test invocation
    if args.test_invoke:
        print("Testing the endpoint with a sample summarization payload...")
        runtime_client = boto3.client("sagemaker-runtime", region_name=args.region)
        payload = {"inputs": "This is a test document. Please generate a concise summary."}
        response = runtime_client.invoke_endpoint(
            EndpointName=args.endpoint_name,
            ContentType="application/json",
            Body=str(payload).encode("utf-8")
        )
        result = response["Body"].read().decode("utf-8")
        print("Inference result:", result)

    print("Done. Your summarization endpoint is ready!")


if __name__ == "__main__":
    main()

