# MLOPS_Project
A MLOPs project that automatically collects news data, classifies articles by length category, and serves predictions through a web interface. Implementation of practices such as CI/CD, data versioning with DVC, containerization with Docker, and deployment on Kubernetes.

## Environment Setup

### Required API Keys
This project requires a News API key for data collection:

1. Get a free API key from [News API](https://newsapi.org/)
2. Set the environment variable:
   ```
   set NEWS_API_KEY=your-api-key-here
   ```
   
   Or copy `.env.template` to `.env` and update the values:
   ```
   cp .env.template .env
   # Edit .env with your API key
   ```

## Running the Project

### Using Docker Compose
```bash
# Set your API key first
set NEWS_API_KEY=your-api-key-here

# Start the services
docker-compose up -d
```

### Using Minikube
```bash
# Deploy to Minikube (will prompt for API key if not set)
deploy-to-minikube.bat

# Set up port forwarding
port-forward.bat
```

## Accessing the Services
- Frontend: http://localhost:80
- Backend API: http://localhost:5000
- MLflow UI: http://localhost:5002
