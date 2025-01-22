#!/usr/bin/env python3

import boto3
import json

def main():
    region = "eu-west-2"
    endpoint_name = "rag-endpoint-minimal-v1"  # Must match what you created
    query_text = "What is the capital of France?"
    top_k = 2

    runtime = boto3.client("runtime.sagemaker", region_name=region)

    payload = {
        "query": query_text,
        "top_k": top_k
    }

    print(f"Invoking endpoint {endpoint_name} with query: {query_text}")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    # The response body is the raw JSON
    result = json.loads(response["Body"].read().decode("utf-8"))
    print("Endpoint response:", result)


if __name__ == "__main__":
    main()

