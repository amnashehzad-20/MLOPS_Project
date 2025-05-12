# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data mlruns models

# Convert Windows line endings to Unix for the entrypoint script
RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh
RUN dos2unix simple-entrypoint.sh && chmod +x simple-entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5001

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
ENTRYPOINT ["/app/entrypoint.sh"]
