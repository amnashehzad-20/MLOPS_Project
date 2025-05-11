pipeline {
    agent any

    environment {
        DOCKER_BACKEND_IMAGE = 'mab3825/mlops-backend:latest'
        DOCKER_FRONTEND_IMAGE = 'mab3825/mlops-frontend:latest'
        DOCKER_CREDENTIALS = 'docker-hub-credentials'  // Defined in Jenkins
        ADMIN_EMAIL = 'i211215@nu.edu.pk'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'dev', url: 'https://github.com/amnashehzad-20/MLOPS_Project.git'
            }
        }

        stage('Build Docker Images') {
            steps {
                bat '''
                docker context use default
                docker build -t %DOCKER_BACKEND_IMAGE% ./backend
                docker build -t %DOCKER_FRONTEND_IMAGE% ./frontend
                '''
            }
        }

        stage('Push Docker Images') {
            steps {
                withDockerRegistry([credentialsId: DOCKER_CREDENTIALS, url: '']) {
                    bat 'docker push %DOCKER_BACKEND_IMAGE%'
                    bat 'docker push %DOCKER_FRONTEND_IMAGE%'
                }
            }
        }

        stage('Deploy to Server') {
            steps {
                bat '''
                docker stop mlops-backend || true
                docker rm mlops-backend || true
                docker run -d --name mlops-backend -p 5000:5000 %DOCKER_BACKEND_IMAGE%
                docker stop mlops-frontend || true
                docker rm mlops-frontend || true
                docker run -d --name mlops-frontend -p 80:80 %DOCKER_FRONTEND_IMAGE%
                '''
            }
        }

    }
}