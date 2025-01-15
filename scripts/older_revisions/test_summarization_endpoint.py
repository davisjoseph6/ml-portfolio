#!/usr/bin/env python3

import boto3
import json

def main():
    region = "eu-west-2"
    endpoint_name = "my-summar-endpoint"  # Must match the name you used in create_endpoint.py

    # 1. Create a SageMaker Runtime client
    runtime_sm = boto3.client("sagemaker-runtime", region_name=region)

    # 2. Prepare your payload
    #    For a Hugging Face summarization model, you typically send JSON like {"inputs": "..."}
    payload = {
        "inputs": (
            "This is an example text that we want to summarize. "
            "The newly deployed summarization model should produce a concise summary of this text."
        )
    }

    # 3. Invoke the SageMaker endpoint
    response = runtime_sm.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    # 4. Parse the response
    response_body = response["Body"].read().decode("utf-8")
    result = json.loads(response_body)

    # 5. Print out the summarization result
    print("----- Summarization Result -----")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

