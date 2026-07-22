import os
import sys
import boto3
from botocore.exceptions import ClientError

def download_production_model():
    # Fetch configurations from environment variables
    # (Avoids hardcoding bucket names or model paths)
    bucket_name = os.getenv("AWS_S3_BUCKET")
    s3_model_key = os.getenv("AWS_MODEL_KEY")
    local_model_path = os.path.join("/app", "models", "model.joblib")  

    # Ensure the local target directory exists inside the container
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

    print(f"Connecting to S3 bucket: {bucket_name}...")
    # Initialize the S3 client. It automatically looks for AWS_ACCESS_KEY_ID 
    # and AWS_SECRET_ACCESS_KEY environment variables.
    s3 = boto3.client('s3')

    try:
        print(f"Downloading {s3_model_key} from S3...")
        s3.download_file(bucket_name, s3_model_key, local_model_path)
        print(f"Success! Model securely downloaded to {local_model_path}")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"AWS Error ({error_code}): {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error downloading model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download_production_model()