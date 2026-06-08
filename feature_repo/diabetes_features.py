import os

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32


patient = Entity(
    name="patient",
    join_keys=["patient_id"],
    description="Unique patient row identifier",
)

# Feature data location is environment-aware: local parquet by default, or an
# S3/Garage object when FEAST_SOURCE_PATH is set (s3://feast/...). The custom
# S3 endpoint (MinIO locally, Garage on the cluster) comes from AWS_ENDPOINT_URL.
FEAST_SOURCE_PATH = os.environ.get(
    "FEAST_SOURCE_PATH",
    "/app/data/processed/diabetes_features.parquet",
)

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
        # Field(name="Cardio_Risk", dtype=Float32),

        # Only engineered feature kept
        # Field(name="metabolic_risk", dtype=Float32),
        # Field(name="health_burden", dtype=Float32),
        # Field(name="age_metabolic_risk", dtype=Float32),
    ],
    source=diabetes_source,
)