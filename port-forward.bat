@echo off
echo Starting port forwarding for all Kubernetes services...

:: Create separate windows for each port-forward command
start "Frontend Port Forward" cmd /k "kubectl port-forward service/frontend-service 3000:80"
start "Backend Port Forward" cmd /k "kubectl port-forward service/backend-service 5000:5000"
start "MLflow Port Forward" cmd /k "kubectl port-forward service/mlflow-service 5001:5001"
start "Airflow Port Forward" cmd /k "kubectl port-forward service/airflow-service 8080:8080"
start "PostgreSQL Port Forward" cmd /k "kubectl port-forward service/postgres-service 5432:5432"

echo.
echo Port forwarding started for:
echo - Frontend: http://localhost:3000
echo - Backend API: http://localhost:5000
echo - MLflow: http://localhost:5001
echo - Airflow: http://localhost:8080
echo - PostgreSQL: localhost:5432
echo.
echo Press any key to stop all port forwarding...
pause > nul

:: Kill all cmd windows with port-forward
taskkill /FI "WindowTitle eq Frontend Port Forward*" /F > nul 2>&1
taskkill /FI "WindowTitle eq Backend Port Forward*" /F > nul 2>&1
taskkill /FI "WindowTitle eq MLflow Port Forward*" /F > nul 2>&1
taskkill /FI "WindowTitle eq Airflow Port Forward*" /F > nul 2>&1
taskkill /FI "WindowTitle eq PostgreSQL Port Forward*" /F > nul 2>&1

echo Port forwarding stopped.
