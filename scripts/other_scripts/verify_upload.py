#!/usr/bin/env python3

import boto3

def list_s3_objects(bucket, prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        for obj in response['Contents']:
            print(obj['Key'])
    else:
        print(f"No objects found in s3://{bucket}/{prefix}")

if __name__ == "__main__":
    # Define the buckets and prefixes to verify
    verifications = [
        {'bucket': 'aym-client-data-in', 'prefix': 'supervised/train/'},
        {'bucket': 'aym-client-data-in', 'prefix': 'supervised/val/'},
        {'bucket': 'aym-client-data-in', 'prefix': 'supervised/test/'},
        {'bucket': 'aym-client-data-test-1', 'prefix': 'supervised/test/'},
        {'bucket': 'aym-client-data-test-2', 'prefix': 'supervised/test/'},
        {'bucket': 'aym-client-data-in', 'prefix': 'unsupervised/train/'},
        {'bucket': 'aym-client-data-in', 'prefix': 'unsupervised/val/'},
        {'bucket': 'aym-client-data-in', 'prefix': 'unsupervised/test/'},
        {'bucket': 'aym-client-data-test-1', 'prefix': 'unsupervised/test/'},
        {'bucket': 'aym-client-data-test-2', 'prefix': 'unsupervised/test/'},
    ]
    
    for verify in verifications:
        print(f"Listing s3://{verify['bucket']}/{verify['prefix']}")
        list_s3_objects(verify['bucket'], verify['prefix'])
        print("\n")

