#!/usr/bin/env python3
import boto3
import json

def main():
    region = "eu-west-2"
    endpoint_name = "rag-endpoint"  # The name you used in deploy_rag_endpoint.py

    # 1) Initialize a sagemaker-runtime client
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    # 2) Prepare a sample JSON payload
    payload = {
        "query": "Where is the shipping_orders doc location?",
        "top_k": 2
    }

    # 3) Invoke the endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(payload),
        ContentType="application/json"
    )

    # 4) Parse and print the result
    body_str = response["Body"].read().decode("utf-8")
    result = json.loads(body_str)
    print("RAG Answer:", result)

if __name__ == "__main__":
    main()

