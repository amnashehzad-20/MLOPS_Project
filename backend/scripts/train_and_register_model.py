"""
Script to check, train and register the news classification model properly in MLflow.
This addresses the error where the model isn't being properly registered.
"""
import os
import sys
import logging
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
logger.info(f"Setting MLflow tracking URI to {mlflow_tracking_uri}")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Model name in MLflow registry
MODEL_NAME = "news_classification_model"

def check_if_model_exists():
    """Check if the model already exists in the MLflow model registry"""
    client = MlflowClient()
    try:
        # Check if model exists in registry
        models = client.search_model_versions(f"name='{MODEL_NAME}'")
        if models:
            logger.info(f"Model '{MODEL_NAME}' exists with {len(models)} versions")
            return True
        else:
            logger.info(f"Model '{MODEL_NAME}' does not exist in model registry")
            return False
    except Exception as e:
        logger.error(f"Error checking model existence: {e}")
        return False

def train_and_register_model():
    """Train and register the model in MLflow"""
    from scripts.model.train_model import train_model_with_mlflow
    
    logger.info("Starting model training and registration process")
    
    # Check if processed data file exists
    if not os.path.exists(os.path.join('data', 'processed_data.csv')):
        logger.error("Processed data file not found. Please run data processing first.")
        return False
    
    # Train and register the model
    result = train_model_with_mlflow()
    
    # Verify model was registered
    if result:
        logger.info("Model trained and registered successfully")
        return True
    else:
        logger.error("Failed to train and register model")
        return False

if __name__ == "__main__":
    # Check if MLflow server is accessible
    try:
        client = MlflowClient()
        client.list_registered_models()
        logger.info("Successfully connected to MLflow server")
    except Exception as e:
        logger.error(f"Failed to connect to MLflow server: {e}")
        sys.exit(1)
    
    # Check if model exists
    if not check_if_model_exists():
        # Train and register model if it doesn't exist
        logger.info("Model not found in registry. Training and registering...")
        if train_and_register_model():
            logger.info("Model training and registration completed successfully")
            sys.exit(0)
        else:
            logger.error("Failed to train and register model")
            sys.exit(1)
    else:
        logger.info("Model already exists in registry")
        sys.exit(0)
