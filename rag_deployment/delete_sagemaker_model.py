#!/usr/bin/env python3

import boto3

def main():
    region = "eu-west-2"       # adjust as needed
    model_name = "rag-model-v1"  # the existing model you want to delete

    sm_client = boto3.client("sagemaker", region_name=region)

    try:
        sm_client.delete_model(ModelName=model_name)
        print(f"Deleted old model: {model_name}")
    except sm_client.exceptions.ClientError as e:
        if "Could not find model" in str(e):
            print(f"Model {model_name} not found, nothing to delete.")
        else:
            raise

if __name__ == "__main__":
    main()

