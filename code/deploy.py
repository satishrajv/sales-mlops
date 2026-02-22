import os
import boto3
import tarfile
import yaml
import json
import shutil
import mlflow
from mlflow.tracking import MlflowClient


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_best_model_run_id(config):
    """Get run_id of champion model from MLflow registry."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Get champion model version
    model_name = config["model"]["name"]
    try:
        # Try to get champion alias
        version = client.get_model_version_by_alias(model_name, "champion")
        run_id = version.run_id
        print(f"Found champion model: version={version.version}, run_id={run_id}")
    except Exception:
        # Fall back to latest version
        versions = client.get_latest_versions(model_name)
        if not versions:
            raise ValueError(f"No versions found for model: {model_name}")
        version = versions[-1]
        run_id = version.run_id
        print(f"No champion alias found. Using latest version={version.version}, run_id={run_id}")

    return run_id


def download_model_artifacts(config, run_id):
    """Download model and label_encoder from S3."""
    s3_client = boto3.client("s3", region_name=config["aws"]["region"])
    bucket = config["aws"]["s3_bucket"]
    experiment_id = "1"

    local_model_dir = "model_artifacts"
    os.makedirs(local_model_dir, exist_ok=True)

    # Download model.pkl
    model_s3_key = f"data/artifacts/{experiment_id}/{run_id}/artifacts/model/model.pkl"
    print(f"Downloading model from s3://{bucket}/{model_s3_key}")
    s3_client.download_file(bucket, model_s3_key, f"{local_model_dir}/model.pkl")

    # Download label_encoder.pkl
    encoder_s3_key = f"data/artifacts/label_encoder.pkl"
    print(f"Downloading label_encoder from s3://{bucket}/{encoder_s3_key}")
    s3_client.download_file(bucket, encoder_s3_key, f"{local_model_dir}/label_encoder.pkl")

    # Copy inference.py into model artifacts
    shutil.copy("code/inference.py", f"{local_model_dir}/inference.py")
    print("Copied inference.py to model artifacts")

    return local_model_dir


def package_model(local_model_dir):
    """Package model artifacts into model.tar.gz for SageMaker."""
    tar_path = "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for file in os.listdir(local_model_dir):
            tar.add(
                os.path.join(local_model_dir, file),
                arcname=file
            )
    print(f"Packaged model into {tar_path}")
    return tar_path


def upload_model_to_s3(config, tar_path):
    """Upload model.tar.gz to S3."""
    s3_client = boto3.client("s3", region_name=config["aws"]["region"])
    bucket = config["aws"]["s3_bucket"]
    s3_key = "model/model.tar.gz"

    print(f"Uploading model.tar.gz to s3://{bucket}/{s3_key}")
    s3_client.upload_file(tar_path, bucket, s3_key)

    model_s3_uri = f"s3://{bucket}/{s3_key}"
    print(f"Model uploaded to: {model_s3_uri}")
    return model_s3_uri


def deploy_to_sagemaker(config, model_s3_uri):
    """Create or update SageMaker endpoint."""
    sm_client = boto3.client("sagemaker", region_name=config["aws"]["region"])

    endpoint_name = config["sagemaker"]["endpoint_name"]
    instance_type = config["sagemaker"]["instance_type"]
    role_arn = "arn:aws:iam::936408601161:role/AmazonSageMakerFullAccess"
    region = config["aws"]["region"]

    # SKLearn container image URI
    sklearn_image = f"341280168497.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

    model_name = f"sales-prediction-model"
    config_name = f"sales-prediction-config"

    # Step 1: Create Model
    print(f"Creating SageMaker model: {model_name}")
    try:
        sm_client.delete_model(ModelName=model_name)
        print(f"Deleted existing model: {model_name}")
    except Exception:
        pass

    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": sklearn_image,
            "ModelDataUrl": model_s3_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": model_s3_uri,
            },
        },
        ExecutionRoleArn=role_arn,
    )
    print(f"SageMaker model created: {model_name}")

    # Step 2: Create Endpoint Config
    print(f"Creating endpoint config: {config_name}")
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=config_name)
        print(f"Deleted existing endpoint config: {config_name}")
    except Exception:
        pass

    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1,
            }
        ],
    )
    print(f"Endpoint config created: {config_name}")

    # Step 3: Create or Update Endpoint
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Updating existing endpoint: {endpoint_name}")
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    except sm_client.exceptions.ClientError:
        print(f"Creating new endpoint: {endpoint_name}")
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

    print(f"Endpoint deployment started: {endpoint_name}")
    print(f"Note: Endpoint takes 5-10 minutes to become InService")
    print(f"Endpoint URL: https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations")

    return endpoint_name


def cleanup(local_model_dir, tar_path):
    """Clean up local files."""
    if os.path.exists(local_model_dir):
        shutil.rmtree(local_model_dir)
    if os.path.exists(tar_path):
        os.remove(tar_path)
    print("Cleaned up local files")


if __name__ == "__main__":
    print("=" * 50)
    print("Starting SageMaker Deployment")
    print("=" * 50)

    config = load_config()

    # Step 1: Get best model run_id from MLflow
    run_id = get_best_model_run_id(config)

    # Step 2: Download model artifacts from S3
    local_model_dir = download_model_artifacts(config, run_id)

    # Step 3: Package model as model.tar.gz
    tar_path = package_model(local_model_dir)

    # Step 4: Upload to S3
    model_s3_uri = upload_model_to_s3(config, tar_path)

    # Step 5: Deploy to SageMaker
    endpoint_name = deploy_to_sagemaker(config, model_s3_uri)

    # Step 6: Cleanup
    cleanup(local_model_dir, tar_path)

    print("=" * 50)
    print(f"Deployment complete!")
    print(f"Endpoint: {endpoint_name}")
    print("=" * 50)
