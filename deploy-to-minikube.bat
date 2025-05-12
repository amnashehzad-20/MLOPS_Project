@echo off
echo Starting Minikube deployment process...

REM Check if Minikube is running
minikube status
if %ERRORLEVEL% NEQ 0 (
    echo Starting Minikube...
    minikube start
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to start Minikube. Exiting.
        exit /b 1
    )
)

REM Set Docker to use Minikube's Docker daemon
echo Setting Docker environment to Minikube...
@FOR /f "tokens=*" %%i IN ('minikube -p minikube docker-env --shell cmd') DO @%%i

REM Build Docker images directly in Minikube
echo Building backend Docker image...
docker build -t mlops-backend:latest .\backend

echo Building frontend Docker image...
docker build -t mlops-frontend:latest .\frontend

REM Apply Kubernetes configurations
echo Deploying to Kubernetes...
kubectl apply -f .\kubernetes\backend.yaml
kubectl apply -f .\kubernetes\frontend.yaml

echo Waiting for deployments to be ready...
kubectl wait --for=condition=available --timeout=300s deployment/backend-deployment
kubectl wait --for=condition=available --timeout=300s deployment/frontend-deployment

echo Deployment completed successfully.
echo Run port-forward.bat to access the services locally.
