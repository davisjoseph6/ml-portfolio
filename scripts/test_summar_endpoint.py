#!/usr/bin/env python3

import boto3
import json

def main():
    region = "eu-west-2"
    endpoint_name = "my-summar-endpoint"  # The name you gave in create_endpoint.py

    # 1. Initialize the sagemaker-runtime client
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    # 2. Prepare a sample payload
    #    The HF inference container expects a JSON with "inputs" for summarization.
    payload = {
        "inputs": "This is a test document about Amazon SageMaker. "
                  "We want to summarize it to confirm our model is working."
    }
    print("Payload to send:", payload)

    # 3. Invoke the endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",      # Must match your container's expected input
        Body=json.dumps(payload),
    )

    # 4. Read the response
    result = json.loads(response['Body'].read().decode("utf-8"))
    print("\nRaw Response:", result)

    # For a Hugging Face summarization task, you typically get something like:
    # [{"summary_text": "..."}]
    if isinstance(result, list) and len(result) > 0:
        print("Summary:", result[0].get("summary_text"))
    else:
        print("Unexpected result format:", result)

if __name__ == "__main__":
    main()

