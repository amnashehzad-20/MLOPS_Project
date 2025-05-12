import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Mock prediction endpoint for testing
    """
    try:
        data = request.json
        logger.info(f"Received prediction request: {data}")
        
        # Mock response
        response = {
            "prediction": "Medium",
            "prediction_code": 1,
            "warning": "This is a mock response - MLflow model not loaded",
            "probabilities": {
                "Short": 0.2,
                "Medium": 0.5,
                "Long": 0.3
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "Service healthy (mock mode)"})

@app.route('/model-info', methods=['GET'])
def model_info():
    """Returns mock model information"""
    return jsonify({
        "model_name": "news_classification_model",
        "version": "mock",
        "status": "Mock",
        "warning": "Running in mock mode without MLflow"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask server on port {port} (mock mode)")
    app.run(host='0.0.0.0', port=port, debug=False)
