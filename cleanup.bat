@echo off
echo Cleaning up MLOPS Project Kubernetes resources...

echo Deleting deployments and services...
kubectl delete -f .\kubernetes\frontend.yaml --ignore-not-found
kubectl delete -f .\kubernetes\backend.yaml --ignore-not-found

echo Resources cleaned up successfully.
echo.
echo To completely stop Minikube, run: minikube stop
