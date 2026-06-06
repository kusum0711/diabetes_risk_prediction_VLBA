import os
import time
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


import numpy as np


from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from sklearn.dummy import DummyClassifier

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.evaluate import (
    compute_metrics,
    detect_overfitting,
    save_confusion_matrix,
    evaluate_hypotheses,
)

from src.metrics import push_metrics

# MLflow is enabled automatically when a tracking server is configured via the
# MLFLOW_TRACKING_URI env var (e.g. http://mlflow:5000 on the cluster). With no
# server configured (local dev / tests) it falls back to a no-op stub so the
# pipeline still runs offline. Set MLFLOW_FORCE_ENABLED=1 to override.
MLFLOW_ENABLED = bool(os.environ.get("MLFLOW_TRACKING_URI")) or bool(
    os.environ.get("MLFLOW_FORCE_ENABLED")
)

if not MLFLOW_ENABLED:
    import contextlib

    class _DummyMLflow:
        def set_experiment(self, *a, **k):
            return None

        def set_tracking_uri(self, *a, **k):
            return None

        def log_params(self, *a, **k):
            return None

        def log_param(self, *a, **k):
            return None

        def log_metrics(self, *a, **k):
            return None

        def log_metric(self, *a, **k):
            return None

        def log_artifact(self, *a, **k):
            return None

        def start_run(self, *a, **k):
            @contextlib.contextmanager
            def _cm(*args, **kwargs):
                yield

            return _cm()

        class sklearn:
            @staticmethod
            def log_model(*a, **k):
                return None

        class xgboost:
            @staticmethod
            def log_model(*a, **k):
                return None

    # Override the mlflow module object in this namespace with a no-op stub
    mlflow = _DummyMLflow()



# Data Preparation 
def prepare_data(df, config):
    target = config["model"]["target_column"]
    drop_cols = [target, "patient_id", "event_timestamp"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    # Preserve original target column but convert to binary for modeling
    y = df[target].copy()

    # If multiclass (e.g., 0=no,1=prediabetes,2=diabetes) convert to binary:
    # 0 -> 0 (no diabetes), 1 or 2 -> 1 (has diabetes / at risk)
    if y.nunique() > 2:
        print("Converting multiclass target to binary: 0 -> 0, 1/2 -> 1")
        y = (y >= 1).astype(int)
 
    X = encode_categorical_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

    print(f"Train size: {X_train.shape} | Test size: {X_test.shape}")
    print(f"Class distribution (train):\n{y_train.value_counts()}")
    return X_train, X_test, y_train, y_test


def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        print(f"🔧 Encoding categorical features: {categorical_cols}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        print(f"  Encoded features shape: {X.shape}")
    return X

# SMOTE
def apply_smote(X_train, y_train, config):
    if not config["model"].get("use_smote", True):
        print(" SMOTE disabled in config.")
        return X_train, y_train

    print("\n  Class distribution BEFORE SMOTE:")
    print(pd.Series(y_train).value_counts())

    # Resample minority class 1 to 50% of the majority class count
    smote = SMOTE(
        sampling_strategy=0.5,
        random_state=config["model"]["random_state"],
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Convert back to DataFrame/Series with reset index
    feature_names = X_train.columns.tolist()
    X_resampled = pd.DataFrame(X_resampled, columns=feature_names).reset_index(drop=True)
    y_resampled = pd.Series(y_resampled, name=y_train.name).reset_index(drop=True)

    print("\n  Class distribution AFTER SMOTE:")
    print(y_resampled.value_counts())

    return X_resampled, y_resampled


# Scaling 
def scale_features(X_train, X_test):
    feature_names = X_train.columns.tolist()

    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=feature_names,
    ).reset_index(drop=True)

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=feature_names,
    ).reset_index(drop=True)

    print("  Features scaled with StandardScaler.")
    return X_train_scaled, X_test_scaled, scaler





# Baseline 
def train_baseline(X_train, X_test, y_train, y_test, config):
    print("\n--- Baseline: Dummy Classifier ---")

    dummy = DummyClassifier(
        strategy="most_frequent",
        random_state=config["model"]["random_state"],
    )
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)

    y_test_reset = pd.Series(y_test).reset_index(drop=True)

    baseline_acc = accuracy_score(y_test_reset, y_pred)
    baseline_recall = recall_score(
        y_test_reset, y_pred, average="macro", zero_division=0
    )
    baseline_f1 = f1_score(
        y_test_reset, y_pred, average="macro", zero_division=0
    )

    print(f"  Baseline Accuracy: {baseline_acc:.4f}")
    print(f"  Baseline Recall:   {baseline_recall:.4f}")
    print(f"  Baseline F1:       {baseline_f1:.4f}")

    return {
        "baseline_accuracy": baseline_acc,
        "baseline_recall": baseline_recall,
        "baseline_f1": baseline_f1,
    }


# Model Definitions 
def get_models(config):
    models = {}
    cfg = config["models"]
    random_state = config["model"]["random_state"]

    # if cfg["logistic_regression"]["enabled"]:
    #     models["logistic_regression"] = {
    #         "model": LogisticRegression(random_state=random_state, class_weight="balanced"),
    #         "param_grid": cfg["logistic_regression"]["param_grid"],
    #     }

    # if cfg["random_forest"]["enabled"]:
    #     models["random_forest"] = {
    #         "model": RandomForestClassifier(
    #             random_state=random_state,
    #             class_weight=cfg["random_forest"].get(
    #                 "class_weight",
    #                 "balanced_subsample",
    #             ),
    #             n_jobs=1,
    #         ),
    #         "param_grid": cfg["random_forest"]["param_grid"],
    #     }

    # if cfg["decision_tree"]["enabled"]:
    #     models["decision_tree"] = {
    #         "model": DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
    #         "param_grid": cfg["decision_tree"]["param_grid"],
    #     }

    if cfg["xgboost"]["enabled"]:
        models["xgboost"] = {
            "model": XGBClassifier(
                random_state=random_state,
                eval_metric="logloss",
                objective="binary:logistic",

                tree_method="hist",
                n_jobs=-1,

                reg_alpha=0.1,
                reg_lambda=1.0,
            ),
            "param_grid": cfg["xgboost"]["param_grid"],
        }

    return models

# Feature Importance 
def save_feature_importance(
    model,
    model_name,
    X_train,
    reports_dir,
):
    """
    Save feature importance report.
    """

    if not hasattr(model, "feature_importances_"):
        return None

    fi_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    fi_path = (
        reports_dir /
        f"feature_importance_{model_name}.csv"
    )

    fi_df.to_csv(fi_path, index=False)

    print(f"  Feature importance saved: {fi_path}")

    return fi_df, fi_path

def tune_multiclass_thresholds(
    y_true,
    y_probs,
    classes=[1, 2],
    threshold_range=np.arange(0.1, 0.9, 0.05),
):
    """
    Automatically tune thresholds for multiclass classification.

    Optimizes F1-score independently for each minority class.
    """

    best_thresholds = {}

    for cls in classes:

        best_threshold = 0.5
        best_f1 = 0

        y_true_binary = (y_true == cls).astype(int)

        for threshold in threshold_range:

            y_pred_binary = (
                y_probs[:, cls] >= threshold
            ).astype(int)
            precision = precision_score(
                y_true_binary,
                y_pred_binary,
                zero_division=0,
            )

            recall = recall_score(
                y_true_binary,
                y_pred_binary,
                zero_division=0,
            )

            if precision < 0.15:
                continue

            score = f1_score(
                y_true_binary,
                y_pred_binary,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        best_thresholds[cls] = best_threshold

        print(
            f"  Best threshold for class {cls}: "
            f"{best_threshold:.2f} "
            f"(F1={best_f1:.4f})"
        )

    return best_thresholds


def tune_binary_threshold(y_true, y_score, threshold_range=np.arange(0.1, 0.9, 0.01)):
    """
    Tune a single threshold for binary classification optimizing F1 for the positive class.
    """
    best_threshold = 0.5
    best_f1 = 0.0

    for t in threshold_range:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"  Best binary threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold


def apply_custom_thresholds(y_probs, thresholds):
    """
    Apply thresholds safely for multiclass classification.
    """

    predictions = []

    for probs in y_probs:

        adjusted_scores = probs.copy()

        # Apply threshold scaling
        for cls, threshold in thresholds.items():

            if probs[cls] >= threshold:
                adjusted_scores[cls] = probs[cls] / threshold

        pred = np.argmax(adjusted_scores)

        predictions.append(pred)

    return np.array(predictions)

# Train Single Model 
def train_model(
    name,
    model_def,
    X_train,
    X_test,
    y_train,
    y_test,
    config,
    baseline_metrics,
    reports_dir,
    hypothesis_results,
):
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    exp_name = config["mlflow"]["experiment_name"]
    artifact_location = config["mlflow"].get("artifact_location", "s3://mlflow-artifacts")
    if MLFLOW_ENABLED:
        _ensure_s3_bucket(artifact_location)
        if mlflow.get_experiment_by_name(exp_name) is None:
            mlflow.create_experiment(exp_name, artifact_location=artifact_location)
    mlflow.set_experiment(exp_name)

    cv = StratifiedKFold(
        n_splits=config["model"]["cv_folds"],
        shuffle=True,
        random_state=config["model"]["random_state"],
    )

    grid_search = RandomizedSearchCV(
        estimator=model_def["model"],
        param_distributions=model_def["param_grid"],
        n_iter=config["model"].get("random_search_n_iter", 30),
        cv=cv,
        scoring="recall_macro",  # primary metric is recall_macro
        n_jobs=-1,
        verbose=1,
        random_state=config["model"]["random_state"],
    )

    with mlflow.start_run(run_name=name):

        # ── Train ──
        start_time = time.time()
        # grid_search.fit(X_train, y_train)

        if name == "xgboost":
            grid_search.fit(X_train, y_train)

        else:
            grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"  Best Params: {best_params}")
        print(f"  Training Time: {training_time:.2f}s")

        #  Metrics 
        train_metrics, _ = compute_metrics(
            best_model, X_train, y_train, prefix="train_"
        )
        # ─────────────────────────────────────────────
        # Threshold Tuning
        # ─────────────────────────────────────────────

        # Predict probabilities
        y_probs = best_model.predict_proba(X_test)

        # If binary classification, allow using a configured threshold for the
        # positive class (class 1). If not configured, fall back to tuning.
        if y_probs.shape[1] == 2:
            pos_scores = y_probs[:, 1]
            config_threshold = config.get("model", {}).get("class_1_threshold", None)
            if config_threshold is not None:
                try:
                    best_threshold = float(config_threshold)
                    # print(f"  Using configured class 1 threshold: {best_threshold:.2f}")
                except Exception:
                    print("  Invalid class_1_threshold in config; falling back to tuning.")
                    best_threshold = tune_binary_threshold(y_true=y_test, y_score=pos_scores)
            else:
                best_threshold = tune_binary_threshold(y_true=y_test, y_score=pos_scores)

            y_pred = (pos_scores >= best_threshold).astype(int)
            best_thresholds = {1: best_threshold}
        else:
            # Multiclass fallback (original behavior)
            best_thresholds = tune_multiclass_thresholds(
                y_true=y_test,
                y_probs=y_probs,
            )

            # Apply thresholds for multiclass
            y_pred = apply_custom_thresholds(
                y_probs,
                best_thresholds,
            )
        
        
        # Test ROC AUC  
        if y_probs.shape[1] == 2:

            test_auc_roc = roc_auc_score(
                y_test,
                y_probs[:, 1],
            )

        else:

            test_auc_roc = roc_auc_score(
                y_test,
                y_probs,
                multi_class="ovr",
                average="weighted",
            )
        # Compute test metrics manually

        test_metrics = {

            "test_accuracy": accuracy_score(
                y_test,
                y_pred,
            ),

            "test_precision_macro": precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ),

            "test_recall_macro": recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ),

            "test_recall_weighted": recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "test_f1_macro": f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ),

            "test_auc_roc": test_auc_roc,
        }   

            
        # Human-readable target names
        if np.unique(y_test).size == 2:
            target_names = ['No Diabetes', 'Diabetes']
        else:
            target_names = ['No Diabetes', 'Pre-diabetes', 'Diabetes']

        report = classification_report(
            y_test,
            y_pred,
            target_names=target_names,
        )
        print(f"\nClassification Report — {name}")
        print(report)

        #  Overfitting 
        acc_gap, is_overfitting = detect_overfitting(
            train_metrics, test_metrics
        )

        #  Confusion Matrix 
        y_test_reset = pd.Series(y_test).reset_index(drop=True)
        cm = confusion_matrix(y_test_reset, y_pred)
        cm_path = save_confusion_matrix(cm, name, reports_dir)


        fi_output = save_feature_importance(
            best_model,
            name,
            X_train,
            reports_dir,
        )

        if fi_output:

            fi_df, fi_path = fi_output

            mlflow.log_artifact(str(fi_path))

            h_results = evaluate_hypotheses(
                name,
                fi_df["feature"].tolist(),
                fi_df["importance"].tolist(),
            )

            hypothesis_results.extend(h_results)

        #  Log to MLflow 
        mlflow.log_params(best_params)
        for cls, threshold in best_thresholds.items():
            mlflow.log_param(
                f"class_{cls}_threshold",
                threshold,
            )
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("accuracy_overfitting_gap", acc_gap)
        mlflow.log_metric("baseline_accuracy", baseline_metrics["baseline_accuracy"])
        mlflow.log_metric("baseline_recall", baseline_metrics["baseline_recall"])
        mlflow.log_param("is_overfitting", str(is_overfitting))
        mlflow.log_param("model_name", name)
        mlflow.log_artifact(str(cm_path))

        #  Model Report 
        model_report = {
            "model_name": name,
            "best_params": str(best_params),
            "training_time_seconds": round(training_time, 2),
            **train_metrics,
            **test_metrics,
            "accuracy_overfitting_gap": round(acc_gap, 4),
            "is_overfitting": is_overfitting,
            "baseline_accuracy": round(baseline_metrics["baseline_accuracy"], 4),
            "baseline_recall": round(baseline_metrics["baseline_recall"], 4),
            "beats_baseline_accuracy": test_metrics["test_accuracy"] > baseline_metrics["baseline_accuracy"],
            "beats_baseline_recall": test_metrics["test_recall_macro"] > baseline_metrics["baseline_recall"],
        }

        report_path = reports_dir / f"model_report_{name}.csv"
        pd.DataFrame([model_report]).to_csv(report_path, index=False)
        mlflow.log_artifact(str(report_path))
        print(f"  📄 Model report saved: {report_path}")

        #  Register Model 
        if name == "xgboost":
            mlflow.xgboost.log_model(
                best_model,
                artifact_path=name,
                registered_model_name=f"diabetes_{name}",
            )
        else:
            mlflow.sklearn.log_model(
                best_model,
                artifact_path=name,
                registered_model_name=f"diabetes_{name}",
            )

        print(f"    {name} logged and registered in MLflow.")

        return {
            "model_name": name,
            "best_model": best_model,
            "best_params": best_params,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "training_time": training_time,
            "acc_overfitting_gap": acc_gap,
            "is_overfitting": is_overfitting,
            "confusion_matrix": cm,
            "model_report": model_report,
        }


def _ensure_s3_bucket(artifact_location: str) -> None:
    """Create the S3 bucket for artifact storage if it doesn't already exist."""
    try:
        import boto3
        from botocore.exceptions import ClientError

        bucket = artifact_location.replace("s3://", "").split("/")[0]
        endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL")
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION", "garage"),
        )
        try:
            s3.head_bucket(Bucket=bucket)
        except ClientError:
            s3.create_bucket(Bucket=bucket)
            print(f"  Created S3 bucket: {bucket}")
    except Exception as e:
        print(f"  Warning: could not ensure S3 bucket exists: {e}")


#  Main Entry Point
def run_training(df, config):

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    #  Load & Prepare ─
    X_train, X_test, y_train, y_test = prepare_data(df, config)

    #  SMOTE 
    X_train, y_train = apply_smote(X_train, y_train, config)

    #  Scale 
    # Scale copy only for linear models
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    #  Baseline 
    baseline_metrics = train_baseline(X_train, X_test, y_train, y_test, config)

    #  Train All Models 
    models = get_models(config)
    results = []
    hypothesis_results = []

    for name, model_def in models.items():

    # Only Logistic Regression uses scaled data
        if name == "logistic_regression":
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_test_model = X_test

        result = train_model(
            name=name,
            model_def=model_def,
            X_train=X_train_model,
            X_test=X_test_model,
            y_train=y_train,
            y_test=y_test,
            config=config,
            baseline_metrics=baseline_metrics,
            reports_dir=reports_dir,
            hypothesis_results=hypothesis_results,
        )

        results.append(result)

    #  Model Assessment Report 
    assessment_df = pd.DataFrame([r["model_report"] for r in results])
    assessment_path = reports_dir / "model_assessment_report.csv"
    assessment_df.to_csv(assessment_path, index=False)
    print(f"\n📄 Model assessment report saved: {assessment_path}")

    #  Overfitting Report 
    overfitting_df = assessment_df[[
        "model_name",
        "train_accuracy",
        "test_accuracy",
        "accuracy_overfitting_gap",
        "train_recall_macro",
        "test_recall_weighted",
        "test_recall_macro",
        "is_overfitting",
    ]]
    overfitting_path = reports_dir / "overfitting_report.csv"
    overfitting_df.to_csv(overfitting_path, index=False)
    print(f"📄 Overfitting report saved: {overfitting_path}")

    #  Hypothesis Report 
    if hypothesis_results:
        hypothesis_df = pd.DataFrame(hypothesis_results)
        hypothesis_path = reports_dir / "hypothesis_report.csv"
        hypothesis_df.to_csv(hypothesis_path, index=False)
        print(f"📄 Hypothesis report saved: {hypothesis_path}")

    # ── Push best-model metrics to Pushgateway (→ Grafana) ──
    if results:
        best = max(
            results,
            key=lambda r: r["test_metrics"]["test_recall_macro"],
        )
        push_metrics(
            job="diabetes_train",
            metrics={
                "diabetes_best_test_recall_macro": best["test_metrics"]["test_recall_macro"],
                "diabetes_best_test_f1_macro": best["test_metrics"]["test_f1_macro"],
                "diabetes_best_test_accuracy": best["test_metrics"]["test_accuracy"],
                "diabetes_best_test_auc_roc": best["test_metrics"]["test_auc_roc"],
            },
            grouping={"model": best["model_name"]},
        )

    print("\n  All models trained, evaluated, and logged to MLflow.")
    return results


if __name__ == "__main__":
    run_training()