Write-Host "Deploying standalone backend and MLflow..."

# Build the standalone backend image
Write-Host "Building standalone backend image..."
docker build -f backend/Dockerfile.standalone -t mlops-backend-standalone:latest ./backend

# Import to minikube
Write-Host "Loading image to minikube..."
minikube image load mlops-backend-standalone:latest

# Deploy MLflow first
Write-Host "Deploying MLflow..."
kubectl apply -f kubernetes/mlflow-standalone.yaml

# Wait for MLflow to be ready
Write-Host "Waiting for MLflow to be ready..."
Start-Sleep -Seconds 10

# Deploy backend
Write-Host "Deploying backend..."
kubectl apply -f kubernetes/backend-standalone.yaml

# Check status
Write-Host "Checking deployment status..."
kubectl get pods -l app=mlflow
kubectl get pods -l app=backend

Write-Host "Deployment complete. Backend is running without model loading."
Write-Host "You need to train and register a model first."
