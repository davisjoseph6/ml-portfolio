#!/usr/bin/env python3

import boto3, json
runtime = boto3.client("sagemaker-runtime", region_name="eu-west-2")
payload = {"inputs": "This is an invoice with shipping details, etc."}
response = runtime.invoke_endpoint(
    EndpointName="distilbert-tf-endpoint",
    ContentType="application/json",
    Body=json.dumps(payload)
    )
result = json.loads(response["Body"].read().decode("utf-8"))
print(result)
