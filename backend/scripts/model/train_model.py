import os
import pandas as pd
import numpy as np
import logging
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model_with_mlflow():
    """
    Train a classification model to predict content length category
    (short, medium, long) based on features in the processed news data,
    and track experiments using MLflow.
    """
    input_path = os.path.join('data', 'processed_data.csv')
    model_path = os.path.join('models', 'model.pkl')
    
    try:
        # Start an MLflow run
        mlflow.set_experiment("news_classification_experiment")
        with mlflow.start_run():
            # Set the experiment name
            mlflow.log_param("input_path", input_path)
            mlflow.log_param("model_path", model_path)
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("model_version", "1.0")
            # Read the processed data
            logger.info(f"Reading processed data from {input_path}")
            df = pd.read_csv(input_path)
            
            if df.empty:
                logger.error("Processed data file is empty.")
                return False
            
            # Get all numeric columns except the target and metadata
            exclude_cols = ['content_length', 'content_length_category', 'processed_date', 
                            'collection_date', 'title', 'author', 'source', 'published_at']
            features = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
            
            logger.info(f"Using features: {features}")
            target = 'content_length_category'
            
            # Check if target column exists, if not, create it
            if target not in df.columns:
                logger.info("Creating content length categories")
                q25 = df['content_length'].quantile(0.33)
                q75 = df['content_length'].quantile(0.66)
                df[target] = pd.cut(
                    df['content_length'], 
                    bins=[float('-inf'), q25, q75, float('inf')], 
                    labels=[0, 1, 2]  # 0=short, 1=medium, 2=long
                ).astype(int)
                logger.info(f"Content length categories distribution: {df[target].value_counts()}")
            
            # Drop rows with missing values
            required_columns = features + [target]
            df = df.dropna(subset=required_columns)
            
            if df.empty:
                logger.error("No valid data rows after dropping missing values.")
                return False
            
            # Create feature matrix X and target vector y
            X = df[features]
            y = df[target]
            
            # Split the data
            logger.info("Splitting data into training and testing sets")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Log parameters to MLflow
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            
            # Create a pipeline with feature selection and model
            logger.info("Setting up model pipeline with feature selection")
            pipeline = Pipeline([
                ('feature_selection', SelectKBest(f_regression, k=min(5, len(features)))),
                ('model', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
            ])
            
            # Log model parameters
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.log_param("k_features", min(5, len(features)))
            
            # Perform cross-validation
            logger.info("Performing cross-validation")
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
            mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
            mlflow.log_metric("cv_accuracy_std", cv_scores.std())
            
            # Train the model
            logger.info("Training final model on all data")
            pipeline.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Log metrics to MLflow
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("f1_weighted", class_report['weighted avg']['f1-score'])
            mlflow.log_metric("precision_weighted", class_report['weighted avg']['precision'])
            mlflow.log_metric("recall_weighted", class_report['weighted avg']['recall'])
            
            # Log the model to MLflow
            mlflow.sklearn.log_model(pipeline, "model")
            
            # Create a visualization of the confusion matrix
            os.makedirs('models', exist_ok=True)
            plt.figure(figsize=(8, 6))
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(set(y)))
            plt.xticks(tick_marks, ['Short', 'Medium', 'Long'])
            plt.yticks(tick_marks, ['Short', 'Medium', 'Long'])
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    plt.text(j, i, str(conf_matrix[i, j]),
                             horizontalalignment="center",
                             color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
                             
            plt.tight_layout()
            confusion_matrix_path = os.path.join('models', 'confusion_matrix.png')
            plt.savefig(confusion_matrix_path)
            plt.close()
            
            # Log confusion matrix as an artifact
            mlflow.log_artifact(confusion_matrix_path)
            
            # Log the model to MLflow
            logger.info("Logging the model to MLflow")
            mlflow.sklearn.log_model(pipeline, "model")
            
            # Register the model in the MLflow Model Registry
            logger.info("Registering the model in MLflow Model Registry")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            model_name = "news_classification_model"
            result = mlflow.register_model(model_uri, model_name)
        
            
            client = MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage="Production",  # or "Production" based on your workflow
            )
            logger.info(f"Model registered as '{model_name}' with version {result.version} and transitioned to 'Production'")
            
            logger.info("Model training, logging, and registration completed successfully.")
            return True
        
    except FileNotFoundError:
        logger.error(f"Processed data file not found at {input_path}")
        return False
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False

if __name__ == "__main__":
    train_model_with_mlflow()