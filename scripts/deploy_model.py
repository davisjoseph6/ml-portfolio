#!/usr/bin/env python3

import sagemaker
from sagemaker.huggingface import HuggingFaceModel

def main():
    # 1) S3 path to your model tar.gz
    model_data_s3 = "s3://aym-client-data-in/distilbert_model.tar.gz"

    # 2) IAM role with SageMaker & S3 permissions
    role = "arn:aws:iam::637423166046:role/service-role/AmazonSageMaker-ExecutionRole-20241216T172309"

    # 3) Create a Hugging Face model object with HF_TASK
    huggingface_model = HuggingFaceModel(
        model_data=model_data_s3,
        role=role,
        transformers_version="4.26",
        tensorflow_version="2.11",
        py_version="py39",
        env={  # <-- Add this
            "HF_TASK": "text-classification"  # or "fill-mask", "question-answering", etc.
        }
    )

    # 4) Deploy the model to a SageMaker endpoint
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name="distilbert-tf-endpoint"
    )

    print("Endpoint is deployed! Endpoint Name:", predictor.endpoint_name)

    # 5) Test the endpoint
    sample_payload = {"inputs": "This is a test from the newly deployed model!"}
    prediction = predictor.predict(sample_payload)
    print("Sample prediction:", prediction)

if __name__ == "__main__":
    main()

