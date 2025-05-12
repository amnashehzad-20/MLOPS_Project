import os
import sys
import subprocess
import time
from scripts.model.check_model import ensure_model_exists

def main():
    """Initialize and run the application."""
    print("Starting MLflow server...")
    # Start MLflow server
    mlflow_process = subprocess.Popen(["mlflow", "server", "--host", "0.0.0.0", "--port", "5001", 
                                      "--backend-store-uri", "sqlite:///mlflow.db", 
                                      "--default-artifact-root", "/app/mlruns"])
    time.sleep(5)
    
    # Check/train model
    print("Checking model status...")
    if not ensure_model_exists():
        print("Failed to initialize model. Exiting.")
        mlflow_process.terminate()
        sys.exit(1)
        
    # Start Flask server
    print("Starting Flask server...")
    from server import app
    app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()