# What it does: Evaluates the trained model on the test set and logs metrics to MLflow

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import yaml
import os
import joblib

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model():
    print("=" * 50)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 50)

    # Load config
    config = load_config()
    tracking_uri = config['mlflow']['tracking_uri']
    experiment_name = config['mlflow']['experiment_name']

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load test data
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    print(f"✅ Loaded test data: {X_test.shape}")

    # Load model
    model = joblib.load('data/model/sales_model.pkl')
    print(f"✅ Model loaded")

    # Load run ID
    with open('data/model/run_id.txt', 'r') as f:
        run_id = f.read().strip()
    print(f"✅ Run ID: {run_id}")

    # Make predictions
    y_pred = model.predict(X_test)
    print(f"✅ Predictions made")

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Model Evaluation Metrics:")
    print(f"   MAE  (Mean Absolute Error):  {mae:.4f}")
    print(f"   MSE  (Mean Squared Error):   {mse:.4f}")
    print(f"   RMSE (Root Mean Sq Error):   {rmse:.4f}")
    print(f"   R2   (R-Squared Score):      {r2:.4f}")

    # Log metrics to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        print(f"\n✅ Metrics logged to MLflow")

    # Evaluation result
    if r2 >= 0.8:
        print(f"\n🎉 Model performance is GOOD (R2={r2:.4f})")
    elif r2 >= 0.6:
        print(f"\n⚠️  Model performance is ACCEPTABLE (R2={r2:.4f})")
    else:
        print(f"\n❌ Model performance is POOR (R2={r2:.4f})")

    print("=" * 50)
    return mae, rmse, r2

if __name__ == "__main__":
    evaluate_model()
