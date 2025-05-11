#!/bin/sh

# Start MLflow server in the background
echo "Starting MLflow server..."
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /app/mlruns &
sleep 5

# Check if model exists, if not train it
python -c "
import mlflow
from mlflow.tracking import MlflowClient
import os
import sys

mlflow.set_tracking_uri('http://localhost:5001')
client = MlflowClient()

try:
    models = client.search_model_versions('name=\"news_classification_model\"')
    if not models:
        print('No model found. Training model...')
        # Check if processed data exists
        if os.path.exists('data/processed_data.csv'):
            from scripts.model.train_model import train_model_with_mlflow
            if train_model_with_mlflow():
                print('Model trained successfully')
            else:
                print('Failed to train model')
                sys.exit(1)
        else:
            print('No processed data found. Running data processing...')
            from scripts.data.load_data import load_raw_data
            from scripts.data.process_data import process_data
            if load_raw_data() and process_data():
                print('Data processing completed. Training model...')
                from scripts.model.train_model import train_model_with_mlflow
                if train_model_with_mlflow():
                    print('Model trained successfully')
                else:
                    print('Failed to train model')
                    sys.exit(1)
            else:
                print('Failed to process data')
                sys.exit(1)
    else:
        print(f'Model found: {len(models)} versions')
except Exception as e:
    print(f'Error checking model: {e}')
    sys.exit(1)
"

# Start the Flask server
echo "Starting Flask server..."
python server.py
