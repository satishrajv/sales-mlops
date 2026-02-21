# Training data (80%) → Model learns from this Testing data  (20%) → We check if model learned correctly


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn
import boto3
import yaml
import os
import pickle
import joblib

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def upload_to_s3(local_path, bucket, s3_path, region):
    """Upload a file to S3"""
    s3_client = boto3.client('s3', region_name=region)
    s3_client.upload_file(local_path, bucket, s3_path)
    print(f"✅ Uploaded {local_path} → s3://{bucket}/{s3_path}")

def train_model():
    print("=" * 50)
    print("STEP 3: MODEL TRAINING")
    print("=" * 50)

    # Load config
    config = load_config()
    tracking_uri = config['mlflow']['tracking_uri']
    experiment_name = config['mlflow']['experiment_name']
    model_name = config['model']['name']
    bucket = config['aws']['s3_bucket']
    artifacts_path = config['aws']['s3_artifacts_path']
    region = config['aws']['region']

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"✅ MLflow tracking URI: {tracking_uri}")
    print(f"✅ Experiment: {experiment_name}")

    # Load processed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    print(f"✅ Loaded training data: {X_train.shape}")

    # Start MLflow run
    with mlflow.start_run() as run:
        print(f"\n🚀 MLflow Run ID: {run.info.run_id}")

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        print(f"✅ Model trained successfully")

        # Log parameters to MLflow
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("training_samples", X_train.shape[0])
        mlflow.log_param("features", X_train.shape[1])
        print(f"✅ Parameters logged to MLflow")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=model_name
        )
        print(f"✅ Model logged to MLflow")

        # Save model locally in runner
        os.makedirs('data/model', exist_ok=True)
        joblib.dump(model, 'data/model/sales_model.pkl')
        print(f"✅ Model saved in runner: data/model/sales_model.pkl")

        # Save run ID for next step
        with open('data/model/run_id.txt', 'w') as f:
            f.write(run.info.run_id)
        print(f"✅ Run ID saved")

        # Upload model to S3 (permanent storage)
        print(f"\n📤 Uploading model to S3...")
        upload_to_s3(
            'data/model/sales_model.pkl',
            bucket,
            f"{artifacts_path}sales_model.pkl",
            region
        )

        # Upload label encoder to S3 (needed for FastAPI later)
        upload_to_s3(
            'data/processed/label_encoder.pkl',
            bucket,
            f"{artifacts_path}label_encoder.pkl",
            region
        )

        # Upload run ID to S3
        upload_to_s3(
            'data/model/run_id.txt',
            bucket,
            f"{artifacts_path}run_id.txt",
            region
        )

        print(f"\n✅ All artifacts uploaded to s3://{bucket}/{artifacts_path}")

    print("=" * 50)
    return model

if __name__ == "__main__":
    train_model()
