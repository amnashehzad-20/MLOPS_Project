#!/bin/sh

# Start MLflow server in the background with reduced workers and gunicorn workers
echo "Starting MLflow server..."
export GUNICORN_CMD_ARGS="--workers=1 --threads=1 --worker-class=sync --worker-connections=10 --max-requests=100 --max-requests-jitter=10"
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /app/mlruns &

# Wait for MLflow server to start properly
echo "Waiting for MLflow server to initialize..."
sleep 15
echo "MLflow server should be ready now"

# Check if model exists, train and register it if needed
echo "Checking for existing model and training if needed..."
python scripts/train_and_register_model.py

# Check if the script was successful
if [ $? -ne 0 ]; then
    echo "Warning: Model training script encountered an error, but we'll continue anyway."
fi

# Start the Flask server with limited workers
echo "Starting Flask server..."
export FLASK_ENV=production
exec python server.py
