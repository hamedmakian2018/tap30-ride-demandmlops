# Taxi Demand Prediction - MLOps Project

## Project Overview
The goal of this project is to predict taxi demand in a specific part of the city over one-hour periods. This end-to-end MLOps project covers designing, developing, and deploying a machine learning solution that closely resembles a production environment.

## Project Steps

### 1. Setting Up Object Storage
- Configure object storage for dataset hosting
- Make data publicly available through an API for later access

### 2. Project Setup & Tooling
- Install Black for code formatting
- Install isort for import organization
- Set up pre-commit to automate checks for git commits

### 3. Logging Implementation
- Implement a consistent logger across the application

### 4. Configuration Management
- Create functionality for reading and parsing YAML configuration files

### 5. Data Ingestion Pipeline
- Download raw demand data from specified URLs
- Split data into train/validation/test sets
- Save processed data as CSV files

### 6. Exploratory Data Analysis
- Analyze time period vs. demand distribution
- Investigate whether time period zero corresponds to 12:00 AM
- Examine demand patterns across different time periods

### 7. Data Processing
- Transform raw taxi demand data into model-ready format
- Implement preprocessing pipeline

### 8. Model Training
- Develop model training component of MLOps pipeline
- Implement training workflow

### 9. Model Persistence
- Persist ML models using Joblib
- Evaluate compression algorithms in Joblib
- Analyze impact on performance and disk usage

### 10. Experiment Tracking with MLflow
- Track and compare multiple model versions
- Use MLflow dashboard for performance evaluation
- Select best performing model

### 11. Project Packaging
- Package project using setuptools
- Enable module importability beyond main code scope

### 12. API Development with FastAPI
- Create accessible APIs for ML model serving
- Implement prediction endpoints

### 13. Docker Containerization
- Build Docker container for the MLOps project
- Configure container environment

### 14. Continuous Integration (CI)
- Implement CI pipeline using GitHub Actions
- Automate Docker image building and pushing to Docker Hub

### 15. Kubernetes Deployment
- Deploy application to Kubernetes cluster
- Finalize production deployment

## Technologies Used
- Python
- MLflow
- FastAPI
- Docker
- Kubernetes
- GitHub Actions
- Joblib
- Pre-commit
- Black/isort
