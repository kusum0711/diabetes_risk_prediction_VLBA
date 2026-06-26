# Diabetes Risk Prediction - MLOps Pipeline

An end-to-end MLOps system for predicting diabetes risk.

## What It Does

This project trains and serves a binary classifier that predicts whether a patient has diabetes or is at risk, based on 21 health indicators.


## Pipeline Flow

```
Raw Data (S3)
    │
    ▼
Data Preparation
(validate, clean, engineer features)
    │
    ▼
Feature Store (Feast)
 ├── Offline Store (S3)      → used for training
 └── Online Store (Postgres) → used for inference
    │
    ▼
Model Training
(train multiple classifiers, select best by recall, log to MLflow)
    │
    ▼
MLflow Model Registry
    │
    ▼
FastAPI Prediction API
    │
    ▼
Prometheus / Grafana Monitoring
```

## Project Components

```
.
├── data/                   # Raw and processed datasets
├── feature_repo/           # Feast feature definitions and configuration
├── src/                    # Data preparation, training, and inference code
├── configs/                # Pipeline and model configuration
├── local/                  # Local infrastructure (Docker Compose)
├── tests/                  # Unit and integration tests
├── reports/                # Generated metrics, plots, and confusion matrices
├── Dockerfile              # Application container image
├── requirements.txt        # Python dependencies
├── Makefile                # Local development commands
└── .gitlab-ci.yml          # GitLab CI/CD pipeline
```

## Tech Stack

| Component | Technology |
|---|---|
| ML Models | scikit-learn (Logistic Regression, Random Forest, Decision Tree), XGBoost |
| Experiment Tracking & Registry | MLflow |
| Feature Store | Feast |
| Prediction API | FastAPI |
| Object Storage | Garage (S3-compatible) |
| Database | PostgreSQL (Feast online store) |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker, Docker Compose |
| CI/CD | GitLab CI |


## CI/CD Stages

```
test -> build-images -> prepare-data -> train -> online-infer
```

1. **test** - lint (flake8) and unit tests (pytest)
2. **build-images** - build and push Docker images to GitLab registry
3. **prepare-data** - clean data, engineer features, push to Feast offline store
4. **train** - train and compare models, register best model in MLflow
5. **online-infer** - materialize Feast online store, validate end-to-end prediction


## Infrastructure (University VM)

- **VM**: `141.44.31.148`
- **MLflow**: `http://mlflow:5000`
- **Feast**: `http://feast:6000`
- **Pushgateway**: `http://pushgateway:9091/pushgateway`
- **Garage (S3)**: `http://garage:3900`


## Running Locally

```bash
make up       # start local infrastructure
make run      # prepare data + train model
make api      # start prediction API at http://localhost:8000
make test     # run tests
```