import os
import logging
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Global scaler to match training data normalization
scaler = None

def load_normalization_params():
    """Load the normalization parameters from processed data"""
    global scaler
    try:
        # Use the actual ranges from the data analysis
        scaler = MinMaxScaler()
        # Based on actual data: title_length (17-210), description_length (0-260)
        sample_data = pd.DataFrame({
            'title_length': [17, 210],
            'description_length': [0, 260]
        })
        scaler.fit(sample_data)
        logger.info("Normalization parameters loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load normalization parameters: {e}")
        # Use default scaler
        scaler = MinMaxScaler()
        scaler.fit(pd.DataFrame({
            'title_length': [0, 100],
            'description_length': [0, 500]
        }))

def load_model():
    """Load the model from MLflow model registry"""
    global scaler
    load_normalization_params()  # Load normalization parameters
    
    client = MlflowClient()
    model_name = "news_classification_model"
    try:
        # First try to get the production model
        production_model = client.get_latest_versions(model_name, stages=["Production"])[0]
        logger.info(f"Loading production model version {production_model.version}")
        model_uri = f"models:/{model_name}/Production"
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.warning(f"Failed to load production model: {e}")
        logger.info("Attempting to load latest model version instead")
        
        # Fallback: get the latest model version
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if not all_versions:
            raise ValueError(f"No model versions found for {model_name}")
        
        latest_version = sorted(all_versions, key=lambda x: int(x.version), reverse=True)[0]
        logger.info(f"Loading latest model version {latest_version.version}")
        model_uri = f"models:/{model_name}/{latest_version.version}"
        return mlflow.sklearn.load_model(model_uri)

# Load model at startup - delayed to allow MLflow server to initialize
model = None

def initialize_model():
    """Initialize model loading with retry logic"""
    global model
    import time
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001'))
            model = load_model()
            logger.info("Model successfully loaded")
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}: Failed to load model: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    logger.error("Failed to load model after all retries")
    return False

# Initialize model loading on server start
@app.before_request
def load_model_if_needed():
    """Load model on first request if not already loaded"""
    global model
    if model is None:
        initialize_model()

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict content length category
    Expects a JSON with article features in the request body
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get data from request
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
        
        # Process time-based features (note: hour_sin is not used by the model)
        if 'publish_hour' in data:
            hour = data['publish_hour']
            transformed_data['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        else:
            transformed_data['hour_cos'] = 0  # Default value
        
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
            # Default values for time features
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
            # Fallback normalization if scaler not available
            transformed_data['title_length'] = min(data.get('title_length', 50) / 100.0, 1.0)
            transformed_data['description_length'] = min(data.get('description_length', 100) / 500.0, 1.0)
        
        # Copy other features
        transformed_data['has_author'] = data.get('has_author', 0)
        transformed_data['source_category'] = data.get('source_category', 0)
        
        # Convert to DataFrame with correct feature order
        input_df = pd.DataFrame([transformed_data])
        
        # Ensure features are in the exact order the model expects
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

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """API endpoint to reload the model from MLflow registry"""
    global model
    try:
        model = load_model()
        return jsonify({"status": "Model reloaded successfully"})
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if model is None:
        return jsonify({"status": "Service running, but model not loaded"}), 200
    return jsonify({"status": "Service healthy"}), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    """Returns information about the loaded model"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    client = MlflowClient()
    model_name = "news_classification_model"
    
    try:
        # Get model version info
        models = client.search_model_versions(f"name='{model_name}'")
        if not models:
            return jsonify({"error": f"No versions found for model {model_name}"}), 404
        
        # Get latest production model
        production_models = [m for m in models if m.current_stage == "Production"]
        if production_models:
            model_info = production_models[0]
            status = "Production"
        else:
            # Get latest model
            model_info = sorted(models, key=lambda x: int(x.version), reverse=True)[0]
            status = "Latest"
        
        return jsonify({
            "model_name": model_name,
            "version": model_info.version,
            "status": status,
            "creation_timestamp": model_info.creation_timestamp,
            "run_id": model_info.run_id
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)