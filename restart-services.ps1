# Script to restart the Docker services
Write-Host "Stopping all running Docker containers..." -ForegroundColor Yellow
docker-compose down

Write-Host "Rebuilding and starting services..." -ForegroundColor Green
docker-compose up -d --build

Write-Host "Services restarted. Check logs with 'docker-compose logs -f backend'" -ForegroundColor Cyan
