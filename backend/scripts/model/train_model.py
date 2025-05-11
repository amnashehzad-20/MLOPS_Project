# train_model.py
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    """
    Train a classification model to predict content length category
    (short, medium, long) based on features in the processed news data.
    """
    input_path = os.path.join('data', 'processed_data.csv')
    model_path = os.path.join('models', 'model.pkl')
    
    try:
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
            # Define boundaries for short, medium, and long articles
            q25 = df['content_length'].quantile(0.33)
            q75 = df['content_length'].quantile(0.66)
            
            # Create the categorical target
            df[target] = pd.cut(
                df['content_length'], 
                bins=[float('-inf'), q25, q75, float('inf')], 
                labels=[0, 1, 2]  # 0=short, 1=medium, 2=long
            )
            
            # Convert to integer for easier handling
            df[target] = df[target].astype(int)
            
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
        
        # Create a pipeline with feature selection and model
        logger.info("Setting up model pipeline with feature selection")
        pipeline = Pipeline([
            ('feature_selection', SelectKBest(f_regression, k=min(5, len(features)))),
            ('model', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
        ])
        
        # Perform cross-validation
        logger.info("Performing cross-validation")
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation accuracy scores: {cv_scores}")
        logger.info(f"Mean CV accuracy score: {cv_scores.mean():.4f}")
        
        # Train the model
        logger.info("Training final model on all data")
        pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Get feature importance
        model = pipeline.named_steps['model']
        feature_selector = pipeline.named_steps['feature_selection']
        selected_features_mask = feature_selector.get_support()
        selected_features = [features[i] for i in range(len(features)) if selected_features_mask[i]]
        
        # Display feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            logger.info("Feature importances:")
            for i, feature in enumerate(selected_features):
                logger.info(f"  {feature}: {importances[i]:.4f}")
        
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
        
        # Add text annotations to the confusion matrix
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, str(conf_matrix[i, j]),
                        horizontalalignment="center",
                        color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
                        
        plt.tight_layout()
        plt.savefig(os.path.join('models', 'confusion_matrix.png'))
        plt.close()
        
        # Save metrics to JSON for DVC
        metrics = {
            "accuracy": float(accuracy),
            "f1_weighted": float(class_report['weighted avg']['f1-score']),
            "precision_weighted": float(class_report['weighted avg']['precision']),
            "recall_weighted": float(class_report['weighted avg']['recall']),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std())
        }
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(pipeline, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return True
        
    except FileNotFoundError:
        logger.error(f"Processed data file not found at {input_path}")
        return False
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False

if __name__ == "__main__":
    train_model()