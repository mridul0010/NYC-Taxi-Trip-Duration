import os
import sys
import boto3
from botocore.exceptions import ClientError

def download_production_artifacts():
    # Fetch configurations from environment variables
    bucket_name = os.getenv("AWS_S3_BUCKET")
    
    # S3 Keys for both artifacts
    s3_model_key = os.getenv("AWS_MODEL_KEY", "models/model.joblib")
    s3_preprocessor_key = os.getenv("AWS_PREPROCESSOR_KEY", "models/preprocessor_osrm.joblib")
    
    # Local paths inside the container
    local_model_path = os.path.join("/app", "models", "model.joblib")  
    local_preprocessor_path = os.path.join("/app", "models", "preprocessor_osrm.joblib")  

    # Ensure the local target directory exists inside the container
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

    # If the image already contains the artifacts, keep them as a local fallback.
    local_artifacts = {
        s3_model_key: local_model_path,
        s3_preprocessor_key: local_preprocessor_path,
    }
    missing_artifacts = [
        (s3_key, local_path)
        for s3_key, local_path in local_artifacts.items()
        if not os.path.exists(local_path)
    ]

    if not missing_artifacts:
        print("Local model artifacts already exist; skipping S3 download.")
        return

    print(f"Connecting to S3 bucket: {bucket_name}...")
    # Initialize the S3 client (reads AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY from env)
    s3 = boto3.client('s3')

    for s3_key, local_path in missing_artifacts:
        try:
            print(f"Downloading {s3_key} from S3...")
            s3.download_file(bucket_name, s3_key, local_path)
            print(f"Success! Artifact securely downloaded to {local_path}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if os.path.exists(local_path):
                print(
                    f"AWS Error ({error_code}) while downloading {s3_key}: {e}. "
                    f"Using existing local artifact at {local_path} instead.",
                    file=sys.stderr,
                )
                continue
            print(f"AWS Error ({error_code}) while downloading {s3_key}: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            if os.path.exists(local_path):
                print(
                    f"Unexpected error downloading {s3_key}: {e}. "
                    f"Using existing local artifact at {local_path} instead.",
                    file=sys.stderr,
                )
                continue
            print(f"Unexpected error downloading {s3_key}: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    download_production_artifacts()