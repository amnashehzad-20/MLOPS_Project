import os
import logging
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model variable
model = None
scaler = None

# Initialize normalization parameters
def load_normalization_params():
    """Load the normalization parameters from processed data"""
    global scaler
    try:
        scaler = MinMaxScaler()
        sample_data = pd.DataFrame({
            'title_length': [17, 210],
            'description_length': [0, 260]
        })
        scaler.fit(sample_data)
        logger.info("Normalization parameters loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load normalization parameters: {e}")
        scaler = MinMaxScaler()
        scaler.fit(pd.DataFrame({
            'title_length': [0, 100],
            'description_length': [0, 500]
        }))

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict content length category"""
    if model is None:
        return jsonify({"error": "Model not loaded - MLflow connection not established"}), 500
    
    try:
        data = request.json
        
        # Validate basic input
        basic_required_features = ['title_length', 'has_author', 'publish_hour', 'publish_day']
        
        missing_features = [feat for feat in basic_required_features if feat not in data]
        if missing_features:
            return jsonify({
                "error": f"Missing required features: {missing_features}"
            }), 400
        
        # The model was trained with these features in this order
        expected_features = ['description_length', 'title_length', 'has_author', 'source_category', 
                           'hour_cos', 'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos']
        
        # Transform the input features to match what the model expects
        transformed_data = {}
        
        # Process time-based features
        if 'publish_hour' in data:
            hour = data['publish_hour']
            transformed_data['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        else:
            transformed_data['hour_cos'] = 0
        
        if 'publish_day' in data:
            day = data['publish_day']
            transformed_data['day_sin'] = np.sin(2 * np.pi * day / 31)
            transformed_data['day_cos'] = np.cos(2 * np.pi * day / 31)
            
            # Weekday calculation
            import datetime
            current_date = datetime.datetime.now()
            try:
                weekday = datetime.datetime(current_date.year, current_date.month, day).weekday()
                transformed_data['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
                transformed_data['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
            except ValueError:
                weekday = 0
                transformed_data['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
                transformed_data['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
        else:
            transformed_data['day_sin'] = 0
            transformed_data['day_cos'] = 1
            transformed_data['weekday_sin'] = 0
            transformed_data['weekday_cos'] = 1
        
        # Add text features with proper normalization using scaler
        text_features = pd.DataFrame({
            'title_length': [data.get('title_length', 50)],
            'description_length': [data.get('description_length', 100)]
        })
        
        if scaler is not None:
            normalized_features = scaler.transform(text_features)
            transformed_data['title_length'] = normalized_features[0][0]
            transformed_data['description_length'] = normalized_features[0][1]
        else:
            transformed_data['title_length'] = min(data.get('title_length', 50) / 100.0, 1.0)
            transformed_data['description_length'] = min(data.get('description_length', 100) / 500.0, 1.0)
        
        # Copy other features
        transformed_data['has_author'] = data.get('has_author', 0)
        transformed_data['source_category'] = data.get('source_category', 0)
        
        # Convert to DataFrame with correct feature order
        input_df = pd.DataFrame([transformed_data])
        input_df = input_df[expected_features]
        
        logger.info(f"Input data for prediction: {input_df.iloc[0].to_dict()}")
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Map numeric prediction to category label
        category_map = {0: "Short", 1: "Medium", 2: "Long"}
        predicted_category = category_map[prediction]
        
        # Get prediction probabilities if available
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(input_df)[0]
            for i, prob in enumerate(probs):
                probabilities[category_map[i]] = round(float(prob), 4)
        
        response = {
            "prediction": predicted_category,
            "prediction_code": int(prediction),
            "warning": "Model is biased towards 'Long' due to imbalanced training data (469 Long, 10 Short, 9 Medium articles)"
        }
        
        if probabilities:
            response["probabilities"] = probabilities
            
        # Add debugging info if requested
        if data.get('debug', False):
            response["debug"] = {
                "normalized_features": input_df.iloc[0].to_dict(),
                "raw_inputs": transformed_data
            }
            
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        logger.exception("Detailed error:")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    mlflow_status = "connected" if model is not None else "disconnected"
    return jsonify({
        "status": "healthy",
        "mlflow_status": mlflow_status,
        "model_loaded": model is not None
    }), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    """Returns information about the loaded model"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "status": "Model loaded successfully",
        "type": type(model).__name__
    })

# Initialize model from MLflow
def initialize_model():
    """Try to connect to MLflow and load model"""
    global model
    
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-service:5001')
    logger.info(f"Attempting to connect to MLflow at: {mlflow_uri}")
    
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()
        
        model_name = "news_classification_model"
        
        try:
            # Try to load the model
            models = client.search_model_versions(f"name='{model_name}'")
            if models:
                latest_version = sorted(models, key=lambda x: int(x.version), reverse=True)[0]
                model_uri = f"models:/{model_name}/{latest_version.version}"
                model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Model loaded successfully: version {latest_version.version}")
                return True
            else:
                logger.warning(f"No model versions found for {model_name}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to connect to MLflow: {e}")
        return False

if __name__ == '__main__':
    # Initialize normalization parameters
    load_normalization_params()
    
    # Try to initialize model
    initialize_model()
    
    # Start the server
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
