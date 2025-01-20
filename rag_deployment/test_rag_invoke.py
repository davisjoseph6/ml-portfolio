#!/usr/bin/env python3
import boto3
import json

def main():
    region = "eu-west-2"
    endpoint_name = "rag-endpoint-v4"  # Must match what you created in deploy_rag_endpoint.py

    # Create a sagemaker-runtime client
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    # Prepare a sample JSON payload
    payload = {
        "query": "Where is the shipping_orders doc location?",
        "top_k": 2
    }

    # Invoke the endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(payload),
        ContentType="application/json"
    )

    # Parse and print the result
    body_str = response["Body"].read().decode("utf-8")
    result = json.loads(body_str)
    print("RAG Answer:", result)

if __name__ == "__main__":
    main()

