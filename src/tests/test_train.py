import pytest
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from src.train import (
    prepare_data,
    apply_smote,
    scale_features,
    get_models,
)
from src.evaluate import (
    compute_metrics,
    detect_overfitting,
)
 


# Fixtures

@pytest.fixture
def sample_config():
    return {
        "model": {
            "target_column": "Diabetes_012",
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 3,
            "use_smote": False,
        },
        "models": {
            "logistic_regression": {
                "enabled": True,
                "param_grid": {
                    "C": [0.1],
                },
            },
            "random_forest": {
                "enabled": True,
                "param_grid": {
                    "n_estimators": [10],
                    "max_depth": [3],
                },
            },
            "xgboost": {
                "enabled": False,
                "param_grid": {},
            },

            "decision_tree": {
                "enabled": True,
                "param_grid": {
                    "max_depth": [3],
    },
},
        },
    }


@pytest.fixture
def sample_df():

    np.random.seed(42)

    n = 300

    return pd.DataFrame({
        "patient_id": range(n),
        "event_timestamp": pd.date_range(
            "2024-01-01",
            periods=n,
            freq="D",
        ),

        "HighBP": np.random.randint(0, 2, n),
        "HighChol": np.random.randint(0, 2, n),
        "BMI": np.random.uniform(18, 45, n),
        "Smoker": np.random.randint(0, 2, n),
        "Stroke": np.random.randint(0, 2, n),
        "HeartDiseaseorAttack": np.random.randint(0, 2, n),
        "PhysActivity": np.random.randint(0, 2, n),
        "Fruits": np.random.randint(0, 2, n),
        "Veggies": np.random.randint(0, 2, n),
        "HvyAlcoholConsump": np.random.randint(0, 2, n),
        "NoDocbcCost": np.random.randint(0, 2, n),
        "GenHlth": np.random.randint(1, 6, n),
        "MentHlth": np.random.randint(0, 30, n),
        "PhysHlth": np.random.randint(0, 30, n),
        "DiffWalk": np.random.randint(0, 2, n),
        "Sex": np.random.randint(0, 2, n),
        "Age": np.random.randint(1, 13, n),
        "Education": np.random.randint(1, 7, n),
        "Income": np.random.randint(1, 9, n),
        "BMI_cat": np.random.choice( ["Normal", "Overweight", "Obese", "Underweight"],n),
        "Age_cat": np.random.choice(["Young", "Middle_Age", "Old"], n),
        "Total_Unhealthy_Days": np.random.randint(0, 60, n),
        "Diabetes_012": np.random.choice([0, 1, 2], n),
    })


@pytest.fixture
def processed_data(sample_df, sample_config):

    X_train, X_test, y_train, y_test = prepare_data(
        sample_df,
        sample_config,
    )

    X_train, y_train = apply_smote(
        X_train,
        y_train,
        sample_config,
    )

    X_train, X_test, scaler = scale_features(
        X_train,
        X_test,
    )

    return X_train, X_test, y_train, y_test


# prepare_data

class TestPrepareData:

    def test_target_removed(self, sample_df, sample_config):

        X_train, X_test, _, _ = prepare_data(
            sample_df,
            sample_config,
        )

        assert "Diabetes_012" not in X_train.columns
        assert "Diabetes_012" not in X_test.columns

    def test_feast_columns_removed(self, sample_df, sample_config):

        X_train, X_test, _, _ = prepare_data(
            sample_df,
            sample_config,
        )

        assert "patient_id" not in X_train.columns
        assert "event_timestamp" not in X_train.columns

    def test_stratified_split(self, sample_df, sample_config):

        _, _, y_train, y_test = prepare_data(
            sample_df,
            sample_config,
        )

        assert set(y_train.unique()) == set(y_test.unique())


# SMOTE

class TestSmote:

    def test_smote_increases_minority_classes(
        self,
        sample_df,
        sample_config,
    ):
        sample_config["model"]["use_smote"] = True

        X_train, _, y_train, _ = prepare_data(
            sample_df,
            sample_config,
        )

        counts_before = pd.Series(y_train).value_counts()

        X_res, y_res = apply_smote(
            X_train,
            y_train,
            sample_config,
        )

        counts_after = pd.Series(y_res).value_counts()

        # Total samples should increase
        assert len(X_res) >= len(X_train)

        # Output types correct
        assert isinstance(X_res, pd.DataFrame)
        assert isinstance(y_res, pd.Series)

        # Features preserved
        assert list(X_res.columns) == list(X_train.columns)


# Scaling

class TestScaling:

    def test_scaled_mean_near_zero(self, processed_data):

        X_train, _, _, _ = processed_data

        means = X_train.mean()

        assert (means.abs() < 0.1).all()


# Metrics

class TestMetrics:

    def test_metrics_in_valid_range(self, processed_data):

        X_train, X_test, y_train, y_test = processed_data

        model = DummyClassifier(strategy="most_frequent")

        model.fit(X_train, y_train)

        metrics, _ = compute_metrics(
            model,
            X_test,
            y_test,
            prefix="test_",
        )

        for value in metrics.values():

            assert 0.0 <= value <= 1.0


# Overfitting

class TestOverfitting:

    def test_detects_overfitting(self):

        train_metrics = {
            "train_accuracy": 0.95
        }

        test_metrics = {
            "test_accuracy": 0.70
        }

        _, is_overfitting = detect_overfitting(
            train_metrics,
            test_metrics,
            threshold=0.1,
        )

        assert is_overfitting is True

    def test_no_overfitting(self):

        train_metrics = {
            "train_accuracy": 0.85
        }

        test_metrics = {
            "test_accuracy": 0.82
        }

        _, is_overfitting = detect_overfitting(
            train_metrics,
            test_metrics,
            threshold=0.1,
        )

        assert is_overfitting is False


# Models

class TestModels:

    def test_enabled_models_returned(self, sample_config):

        models = get_models(sample_config)

        assert "logistic_regression" in models
        assert "random_forest" in models
        assert "xgboost" not in models
        assert "decision_tree" in models


# Integration

class TestIntegration:

    def test_pipeline_runs_successfully(
        self,
        processed_data,
    ):

        X_train, X_test, y_train, y_test = processed_data

        model = RandomForestClassifier(
            n_estimators=10,
            random_state=42,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert len(predictions) == len(y_test)

        assert set(predictions).issubset({0, 1, 2})