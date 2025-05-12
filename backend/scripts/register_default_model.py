"""
Script to create and register a basic model in MLflow
Run this script to create a default news classification model
"""
import os
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Set the MLflow tracking URI
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001'))

# The name of the model in the MLflow registry
MODEL_NAME = "news_classification_model"

# Create a simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create some sample training data - features should match our expected input
# [description_length, title_length, has_author, source_category, 
#  hour_cos, day_sin, day_cos, weekday_sin, weekday_cos]
X_train = np.array([
    [0.2, 0.3, 1, 0, 0.5, 0.1, 0.9, 0.3, 0.7],  # Sample 1 - Short
    [0.5, 0.6, 1, 1, 0.6, 0.2, 0.8, 0.4, 0.6],  # Sample 2 - Medium
    [0.8, 0.7, 0, 2, 0.7, 0.3, 0.7, 0.5, 0.5],  # Sample 3 - Long
    [0.9, 0.8, 1, 0, 0.8, 0.4, 0.6, 0.6, 0.4],  # Sample 4 - Long
    [0.7, 0.4, 0, 1, 0.9, 0.5, 0.5, 0.7, 0.3],  # Sample 5 - Medium
    [0.3, 0.2, 1, 2, 1.0, 0.6, 0.4, 0.8, 0.2],  # Sample 6 - Short
    [0.9, 0.9, 1, 0, 0.1, 0.7, 0.3, 0.9, 0.1],  # Sample 7 - Long
    [0.8, 0.8, 0, 1, 0.2, 0.8, 0.2, 1.0, 0.0],  # Sample 8 - Long
    [0.4, 0.5, 1, 2, 0.3, 0.9, 0.1, 0.1, 0.9]   # Sample 9 - Medium
])

# Target: 0 = Short, 1 = Medium, 2 = Long
y_train = np.array([0, 1, 2, 2, 1, 0, 2, 2, 1])

# Train the model
model.fit(X_train, y_train)

# Log and register the model with MLflow
with mlflow.start_run(run_name="default_classification_model") as run:
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    
    # Log metrics
    mlflow.log_metric("accuracy", 1.0)  # This is just a placeholder
    
    # Register the model
    mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
    
    print(f"Model registered as '{MODEL_NAME}' (Run ID: {run.info.run_id})")
