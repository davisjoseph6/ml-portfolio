#!/usr/bin/env python3

import boto3

sm = boto3.client("sagemaker")
endpoint_name = "distilbert-tf-endpoint"

# 1) Delete endpoint
sm.delete_endpoint(EndpointName=endpoint_name)

# 2) Delete endpoint config
sm.delete_endpoint_config(EndpointConfigName=endpoint_name)

# 3) (Optional) Delete the model if you want
#    The model name is usually auto-generated, e.g. huggingface-tensorflow-inference-...
#    or see your logs for the exact model name
sm.delete_model(ModelName="huggingface-tensorflow-inference-2025-01-08-05-28-09-549")

