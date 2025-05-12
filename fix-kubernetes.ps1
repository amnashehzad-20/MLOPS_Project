Write-Host "Fixing MLOps Kubernetes deployment..."

# Delete existing deployments
Write-Host "Deleting existing deployments..."
kubectl delete deployment backend-deployment mlflow-deployment

# Build the simple backend image
Write-Host "Building simple backend image..."
docker build -f backend/Dockerfile.simple -t mlops-backend-simple:latest ./backend

# Import to minikube
Write-Host "Loading image to minikube..."
minikube image load mlops-backend-simple:latest

# Apply fixed configurations
Write-Host "Applying fixed MLflow deployment..."
kubectl apply -f kubernetes/mlflow-deployment-fixed.yaml

# Wait for MLflow to be ready
Write-Host "Waiting for MLflow to be ready..."
kubectl wait --for=condition=ready pod -l app=mlflow --timeout=300s

# Apply backend deployment
Write-Host "Applying simple backend deployment..."
kubectl apply -f kubernetes/backend-deployment-simple.yaml

Write-Host "Deployment fixed. Checking status..."
kubectl get pods
