@echo off
echo Starting port forwarding for MLOPS Project...

REM Check if deployments exist
kubectl get deployment backend-deployment >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Backend deployment not found. Run deploy-to-minikube.bat first.
    exit /b 1
)

kubectl get deployment frontend-deployment >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Frontend deployment not found. Run deploy-to-minikube.bat first.
    exit /b 1
)

echo Starting port forwarding in separate windows...

REM Start backend API port forwarding
start "Backend API" cmd /c "kubectl port-forward service/backend-service 5000:5000 && pause"

REM Start MLflow UI port forwarding
start "MLflow UI" cmd /c "kubectl port-forward service/backend-service 5002:5001 && pause"

REM Start frontend port forwarding
start "Frontend" cmd /c "kubectl port-forward service/frontend-service 80:80 && pause"

echo Port forwarding started.
echo.
echo Services available at:
echo  - Frontend UI: http://localhost:80
echo  - Backend API: http://localhost:5000
echo  - MLflow UI:   http://localhost:5002
echo.
echo Close the terminal windows to stop port forwarding.
