# Introduction
In this project, you will work inside a dedicated virtual machine that contains all the tools and services needed to complete your machine learning task. The provided workflow is an example, which can be extended. It is very important to follow to adopt the MLOps process model, which we will discuss throughout the semester.

# Infrastructure setup of your project and access credentials
  - Virtual machine IP: `141.44.31.148`
  - User Name: `mlops-team-4`
  - Password: `uoEsSUKseYrEaLH6u7yiaxsR0oHL9K`
  - Grafana User-name: `admin`
  - Grafana Password: `uoEsSUKseYrEaLH6u7yiaxsR0oHL9K`
  - `ssh -N -v -L 80:localhost:80 -L 9000:localhost:9000 YOUR_USER_NAME@YOUR_IP`

# Access to services
Your services are running and can be accessible by your ci/cd pipline using these addresses:

- `PUSHGATEWAY_URL`: `"http://pushgateway:9091/pushgateway"`
- `MLFLOW_TRACKING_URI`: `"http://mlflow:5000"`
- `FEAST_URL`: `"http://feast:6000"`
- `AWS_ENDPOINT_URL`: `"http://garage:3900"`
- `AWS_ENDPOINT_URL_S3`: `"http://garage:3900"`

# Example workflow 
That is an example workflow, which you can extend based on your use-case to address your machine learning problem following MLOps process model.
workflow:
```
stages:
  - build-images
  - data
  - train
  - infer
  - online-infer
  - extend-data
  - extend-train
  - online-infer-v2

variables:
- `DOCKER_BUILDKIT`: `"1"`
- `BUILDKIT_PROGRESS`: `"plain"`
- `DATASET_IMAGE`: `"${CI_REGISTRY_IMAGE}/YOU_DATA_GENERATOR:${CI_COMMIT_SHORT_SHA}"`
- `YOUR_ML_ALGORITHM_IMAGE`: `"${CI_REGISTRY_IMAGE}/YOUR_ML_ALGORITHM:${CI_COMMIT_SHORT_SHA}"`
- `FEAST_IMAGE`: `"${CI_REGISTRY_IMAGE}/feast:${CI_COMMIT_SHORT_SHA}"`
- `PUSHGATEWAY_URL`: `"http://pushgateway:9091/pushgateway"`
- `MLFLOW_TRACKING_URI`: `"http://mlflow:5000"`
- `FEAST_URL`: `"http://feast:6000"`
- `S3_ENDPOINT`: `"http://garage:3900"`
- `AWS_ENDPOINT_URL`: `"http://garage:3900"`
- `AWS_ENDPOINT_URL_S3`: `"http://garage:3900"`
- `AWS_REGION`: `"garage"`
- `AWS_DEFAULT_REGION`: `"garage"`
- `AWS_ACCESS_KEY_ID`: `""` # Note: Your key is added to your project as a variable and your ci/cd pipline will have access to it.
- `AWS_SECRET_ACCESS_KEY`: `""` # Note: Your key is added to your project as a variable and your ci/cd pipline will have access to it.
- `POSTGRES_FEAST_PASSWORD`: `""` # Note: Your key is added to your project as a variable and your ci/cd pipline will have access to it.
```