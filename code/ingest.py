# What it does: Reads sales_data.csv from your S3 bucket and saves it locally for next steps

import boto3
import pandas as pd
import yaml
import os
import io


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ingest_data():
    print("=" * 50)
    print("STEP 1: DATA INGESTION")
    print("=" * 50)

    # Load config
    config = load_config()
    bucket = config['aws']['s3_bucket']
    s3_path = config['aws']['s3_data_path']
    region = config['aws']['region']

    print(f"✅ Config loaded")
    print(f"✅ S3 Bucket: {bucket}")
    print(f"✅ S3 Path: {s3_path}")

    # Connect to S3
    s3_client = boto3.client('s3', region_name=region)
    print(f"✅ Connected to S3")

    # Download data from S3
    response = s3_client.get_object(Bucket=bucket, Key=s3_path)
    df = pd.read_csv(io.BytesIO(response['Body'].read()))
    print(f"✅ Downloaded data from S3")

    # Print basic info
    print(f"\n📊 Data Shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    print(f"\n📊 Sample Data:")
    print(df.head())
    print(f"\n📊 Data Types:")
    print(df.dtypes)
    print(f"\n📊 Missing Values:")
    print(df.isnull().sum())

    # Save locally
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/sales_data.csv', index=False)
    print(f"\n✅ Data saved locally to data/raw/sales_data.csv")
    print("=" * 50)

    return df

if __name__ == "__main__":
    ingest_data()
