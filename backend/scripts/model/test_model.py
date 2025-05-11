# test_model.py
import os
import pandas as pd
import numpy as np
import logging
import json
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model():
    """
    Test the trained classification model on a test set and evaluate its performance.
    """
    model_path = os.path.join('models', 'model.pkl')
    input_path = os.path.join('data', 'processed_data.csv')
    test_metrics_path = os.path.join('metrics_test.json')
    
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
            
        # Load the model
        logger.info(f"Loading model from {model_path}")
        pipeline = joblib.load(model_path)
        
        # Load processed data
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        if df.empty:
            logger.error("Data file is empty.")
            return False
        
        # Get all numeric columns except the target and metadata
        exclude_cols = ['content_length', 'content_length_category', 'processed_date', 
                         'collection_date', 'title', 'author', 'source', 'published_at']
        features = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        logger.info(f"Using features: {features}")
        target = 'content_length_category'
        
        # Check if target column exists, if not, create it (same as in train_model.py)
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
        
        # Use proper train/test split
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # Ensure y_test and y_pred are integers
        y_test = y_test.astype(int)
        # Make predictions
        logger.info("Making predictions on test data")
        y_pred = pipeline.predict(X_test)
        y_pred = y_pred.astype(int)
        
        # Evaluate model with classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Save test metrics
        metrics = {
            "test_accuracy": float(accuracy),
            "test_f1_weighted": float(class_report['weighted avg']['f1-score']),
            "test_precision_weighted": float(class_report['weighted avg']['precision']),
            "test_recall_weighted": float(class_report['weighted avg']['recall'])
        }
        
        with open(test_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"Test metrics saved to {test_metrics_path}")
        
        # Create a comprehensive visualization
        os.makedirs('models', exist_ok=True)
        plt.figure(figsize=(12, 10))
        
        # 1. Confusion Matrix
        plt.subplot(2, 2, 1)
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(set(y_test)))
        unique_labels = sorted(set(y_test))
        tick_labels = ['Short', 'Medium', 'Long'][:len(unique_labels)]  # Adjust labels based on unique categories
        tick_marks = np.arange(len(unique_labels))

        plt.xticks(tick_marks, tick_labels)
        plt.yticks(tick_marks, tick_labels)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add text annotations
                
        # Add text annotations
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, str(conf_matrix[i, j]),
                        horizontalalignment="center",
                        color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
        # 2. Class Distribution
        plt.subplot(2, 2, 2)
        class_counts = df[target].value_counts().sort_index()
        plt.bar(["Short", "Medium", "Long"], class_counts.values)
        plt.title('Class Distribution')
        plt.ylabel('Count')
        
        # 3. Feature Importance (if available)
        plt.subplot(2, 2, 3)
        if hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
            model = pipeline.named_steps['model']
            if hasattr(model, 'feature_importances_'):
                
                # Get the selected features if feature selection was used
                if 'feature_selection' in pipeline.named_steps:
                    feature_selector = pipeline.named_steps['feature_selection']
                    selected_features_mask = feature_selector.get_support()
                    selected_features = [features[i] for i in range(len(features)) if selected_features_mask[i]]
                    importances = model.feature_importances_
                else:
                    selected_features = features
                    importances = model.feature_importances_
                
                # Sort features by importance
                indices = np.argsort(importances)
                plt.barh(range(len(indices)), importances[indices], align='center')
                plt.yticks(range(len(indices)), [selected_features[i] for i in indices])
                plt.xlabel('Relative Importance')
                plt.title('Feature Importance')
            else:
                plt.text(0.5, 0.5, "Feature importance not available", 
                         horizontalalignment='center', verticalalignment='center')
        else:
            plt.text(0.5, 0.5, "Feature importance not available", 
                     horizontalalignment='center', verticalalignment='center')
        
        # 4. Metrics Summary
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        # Safely access class-wise F1 scores
        short_f1 = class_report.get('0', {}).get('f1-score', 0.0)
        medium_f1 = class_report.get('1', {}).get('f1-score', 0.0)
        long_f1 = class_report.get('2', {}).get('f1-score', 0.0)
        
        plt.text(0.1, 0.9, f"Accuracy: {accuracy:.4f}", fontsize=12)
        plt.text(0.1, 0.8, f"F1 Score (weighted): {class_report['weighted avg']['f1-score']:.4f}", fontsize=12)
        plt.text(0.1, 0.7, f"Precision (weighted): {class_report['weighted avg']['precision']:.4f}", fontsize=12)
        plt.text(0.1, 0.6, f"Recall (weighted): {class_report['weighted avg']['recall']:.4f}", fontsize=12)
        plt.text(0.1, 0.4, "Class-wise F1 Scores:", fontsize=12)
        plt.text(0.1, 0.3, f"Short: {short_f1:.4f}", fontsize=12)
        plt.text(0.1, 0.2, f"Medium: {medium_f1:.4f}", fontsize=12)
        plt.text(0.1, 0.1, f"Long: {long_f1:.4f}", fontsize=12)
        plt.title('Model Performance Metrics')
        
        plt.tight_layout()
        plot_path = os.path.join('models', 'classification_evaluation.png')
        plt.savefig(plot_path)
        logger.info(f"Evaluation plots saved to {plot_path}")
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    test_model()