# Script to restart the MLOps services and fix the MLflow model issue
Write-Host "Stopping all running Docker containers..." -ForegroundColor Yellow
docker-compose down

# Make entrypoint.sh executable
Write-Host "Ensuring entrypoint.sh is executable..." -ForegroundColor Yellow
docker run --rm -v ${PWD}/backend:/app alpine chmod +x /app/entrypoint.sh

Write-Host "Rebuilding and starting services..." -ForegroundColor Green
docker-compose up -d --build

Write-Host "Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "Services restarted. Displaying backend logs..." -ForegroundColor Cyan
docker-compose logs -f backend
