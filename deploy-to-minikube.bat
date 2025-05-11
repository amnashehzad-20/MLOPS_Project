@echo off
echo Starting Minikube deployment...

:: Check if Minikube is running
minikube status > nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Minikube...
    minikube start --memory=8192 --cpus=4
)

:: Set Docker environment for Minikube
echo Setting Docker environment for Minikube...
@FOR /f "tokens=*" %%i IN ('minikube docker-env --shell cmd') DO @%%i

:: Build images locally in Minikube's Docker environment
echo Building Docker images...
docker build -t mlops-backend:latest ./backend
docker build -t mlops-frontend:latest ./frontend

:: Apply Kubernetes configurations
echo Applying Kubernetes configurations...
kubectl apply -f kubernetes/postgres-deployment.yaml
echo Waiting for PostgreSQL to be ready...
timeout /t 20 /nobreak > nul

kubectl apply -f kubernetes/mlflow-deployment.yaml
kubectl apply -f kubernetes/airflow-deployment.yaml
kubectl apply -f kubernetes/backend-deployment.yaml
kubectl apply -f kubernetes/frontend-deployment.yaml

:: Wait for all deployments to be ready
echo Waiting for deployments to be ready...
kubectl wait --for=condition=available --timeout=300s deployment/postgres-deployment
kubectl wait --for=condition=available --timeout=300s deployment/mlflow-deployment
kubectl wait --for=condition=available --timeout=300s deployment/airflow-deployment
kubectl wait --for=condition=available --timeout=300s deployment/backend-deployment
kubectl wait --for=condition=available --timeout=300s deployment/frontend-deployment

echo Deployment completed!
echo.
echo All services use ClusterIP. Run port-forward.bat to access them locally:
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8000
echo MLflow: http://localhost:5000
echo Airflow: http://localhost:8080
echo PostgreSQL: localhost:5432
