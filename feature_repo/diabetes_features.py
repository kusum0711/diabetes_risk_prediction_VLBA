import os

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32


patient = Entity(
    name="patient",
    join_keys=["patient_id"],
    description="Unique patient row identifier",
)

# Feature data location is environment-aware: local parquet by default, or an
# S3/Garage object when FEAST_SOURCE_PATH is set (s3://feast/...). The custom
# S3 endpoint (MinIO locally, Garage on the cluster) comes from AWS_ENDPOINT_URL.
#
# Default resolves relative to this file (feature_repo/../data/...) so it works
# both inside Docker (/app/feature_repo/...) and from any local working directory.
_DEFAULT_PARQUET = str(
    (Path(__file__).parent.parent / "data/processed/diabetes_features.parquet").resolve()
)
FEAST_SOURCE_PATH = os.environ.get("FEAST_SOURCE_PATH", _DEFAULT_PARQUET)

diabetes_source = FileSource(
    path=FEAST_SOURCE_PATH,
    timestamp_field="event_timestamp",
    s3_endpoint_override=os.environ.get("AWS_ENDPOINT_URL"),
)

diabetes_feature_view = FeatureView(
    name="diabetes_features",
    entities=[patient],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="HighBP", dtype=Float32),
        Field(name="HighChol", dtype=Float32),
        Field(name="BMI", dtype=Float32),
        Field(name="Smoker", dtype=Float32),
        Field(name="Stroke", dtype=Float32),
        Field(name="HeartDiseaseorAttack", dtype=Float32),
        Field(name="PhysActivity", dtype=Float32),
        Field(name="Fruits", dtype=Float32),
        Field(name="Veggies", dtype=Float32),
        Field(name="HvyAlcoholConsump", dtype=Float32),
        Field(name="NoDocbcCost", dtype=Float32),
        Field(name="GenHlth", dtype=Float32),
        Field(name="MentHlth", dtype=Float32),
        Field(name="PhysHlth", dtype=Float32),
        Field(name="DiffWalk", dtype=Float32),
        Field(name="Sex", dtype=Float32),
        Field(name="Age", dtype=Float32),
        Field(name="Education", dtype=Float32),
        Field(name="Income", dtype=Float32),
        Field(name="age_metabolic_risk", dtype=Float32),
        Field(name="HealthRiskScore", dtype=Float32),
    ],
    source=diabetes_source,
)

diabetes_feature_service = FeatureService(
    name="diabetes_feature_service",
    features=[diabetes_feature_view],
)
