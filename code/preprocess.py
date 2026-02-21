# What it does: Cleans and prepares the data so the model can understand it


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import yaml
import os
import pickle

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_data():
    print("=" * 50)
    print("STEP 2: PREPROCESSING")
    print("=" * 50)

    # Load config
    config = load_config()
    target_column = config['model']['target_column']
    test_size = config['model']['test_size']
    random_state = config['model']['random_state']

    # Load raw data
    df = pd.read_csv('data/raw/sales_data.csv')
    print(f"✅ Loaded raw data: {df.shape}")

    # Encode categorical column - product_category
    print(f"\n🔄 Encoding categorical columns...")
    le = LabelEncoder()
    df['product_category_encoded'] = le.fit_transform(df['product_category'])
    print(f"✅ Encoded product_category: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Save label encoder for later use
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print(f"✅ Label encoder saved")

    # Define features and target
    feature_columns = ['store_id', 'product_category_encoded',
                       'units_sold', 'unit_price',
                       'discount_pct', 'is_weekend']
    X = df[feature_columns]
    y = df[target_column]

    print(f"\n📊 Features: {feature_columns}")
    print(f"📊 Target: {target_column}")
    print(f"📊 X shape: {X.shape}")
    print(f"📊 y shape: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    print(f"\n✅ Train set: {X_train.shape}")
    print(f"✅ Test set: {X_test.shape}")

    # Save processed data
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    print(f"\n✅ Processed data saved to data/processed/")
    print("=" * 50)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()
