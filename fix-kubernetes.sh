#!/bin/bash

echo "Fixing MLOps Kubernetes deployment..."

# Delete existing deployments
echo "Deleting existing deployments..."
kubectl delete deployment backend-deployment mlflow-deployment

# Build the simple backend image
echo "Building simple backend image..."
docker build -f backend/Dockerfile.simple -t mlops-backend-simple:latest ./backend

# Import to minikube
echo "Loading image to minikube..."
minikube image load mlops-backend-simple:latest

# Apply fixed configurations
echo "Applying fixed MLflow deployment..."
kubectl apply -f kubernetes/mlflow-deployment-fixed.yaml

# Wait for MLflow to be ready
echo "Waiting for MLflow to be ready..."
kubectl wait --for=condition=ready pod -l app=mlflow --timeout=300s

# Apply backend deployment
echo "Applying simple backend deployment..."
kubectl apply -f kubernetes/backend-deployment-simple.yaml

echo "Deployment fixed. Checking status..."
kubectl get pods
