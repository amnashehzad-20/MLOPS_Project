import mlflow
from mlflow.tracking import MlflowClient
import os
import sys

def ensure_model_exists():
    """Check if model exists, if not train it."""
    mlflow.set_tracking_uri('http://localhost:5001')
    client = MlflowClient()

    try:
        models = client.search_model_versions('name="news_classification_model"')
        if not models:
            print('No model found. Training model...')
            # Check if processed data exists
            if os.path.exists('data/processed_data.csv'):
                from scripts.model.train_model import train_model_with_mlflow
                if train_model_with_mlflow():
                    print('Model trained successfully')
                    return True
                else:
                    print('Failed to train model')
                    return False
            else:
                print('No processed data found. Running data processing...')
                from scripts.data.collect_data import collect_news_data
                from scripts.data.preprocess_data import preprocess_data
                if collect_news_data() and preprocess_data():
                    print('Data processing completed. Training model...')
                    from scripts.model.train_model import train_model_with_mlflow
                    if train_model_with_mlflow():
                        print('Model trained successfully')
                        return True
                    else:
                        print('Failed to train model')
                        return False
                else:
                    print('Failed to process data')
                    return False
        else:
            print(f'Model found: {len(models)} versions')
            return True
    except Exception as e:
        print(f'Error checking model: {e}')
        return False