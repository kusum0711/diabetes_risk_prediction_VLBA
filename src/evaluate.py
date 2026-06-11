from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)
import pandas as pd
import matplotlib.pyplot as plt


def compute_metrics(model, X, y, prefix=""):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    y = pd.Series(y).reset_index(drop=True)

    print(f"\n  Classification Report ({prefix})")
    print(
        classification_report(
            y,
            y_pred,
            digits=4,
            zero_division=0,
        )
    )

    # Per-class arrays (index = class label)
    per_class_precision = precision_score(y, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y, y_pred, average=None, zero_division=0)

    metrics = {
        # Overall accuracy
        f"{prefix}accuracy": accuracy_score(y, y_pred),

        # Weighted averages
        f"{prefix}precision_weighted": precision_score(
            y, y_pred, average="weighted", zero_division=0
        ),
        f"{prefix}recall_weighted": recall_score(
            y, y_pred, average="weighted", zero_division=0
        ),
        f"{prefix}f1_weighted": f1_score(
            y, y_pred, average="weighted", zero_division=0
        ),

        # Macro recall
        f"{prefix}recall_macro": recall_score(
            y, y_pred, average="macro", zero_division=0
        ),

        # ROC AUC (binary: positive-class scores; multiclass: OVR weighted)
        f"{prefix}auc_roc": (
            roc_auc_score(y, y_proba[:, 1])
            if (hasattr(y_proba, 'shape') and y_proba.ndim == 2 and y_proba.shape[1] == 2)
            else roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
        ),
    }

    # Per-class precision / recall / f1
    for i, (p, r, f) in enumerate(zip(per_class_precision, per_class_recall, per_class_f1)):
        metrics[f"{prefix}precision_class_{i}"] = float(p)
        metrics[f"{prefix}recall_class_{i}"] = float(r)
        metrics[f"{prefix}f1_class_{i}"] = float(f)

    return metrics, y_pred


# Overfitting Detection

def detect_overfitting(train_metrics, test_metrics, threshold=0.1):

    train_acc = train_metrics["train_accuracy"]
    test_acc = test_metrics["test_accuracy"]

    acc_gap = train_acc - test_acc

    is_overfitting = (
        acc_gap > threshold
    )

    print(
        f"  Train Accuracy: {train_acc:.4f} | "
        f"Test Accuracy: {test_acc:.4f} | "
        f"Gap: {acc_gap:.4f}"
    )

    if is_overfitting:
        print(
            f"  ⚠️ Overfitting detected! "
            f"Gap exceeds threshold ({threshold})"
        )
    else:
        print("    No significant overfitting detected.")

    return acc_gap, is_overfitting


# Confusion Matrix
def save_confusion_matrix(cm, model_name, reports_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    # Choose labels based on confusion matrix shape
    n_labels = cm.shape[0]
    if n_labels == 2:
        labels = ["No Diabetes (0)", "Diabetes (1)"]
    else:
        labels = ["No Diabetes (0)", "Pre Diabetes (1)", "Diabetes (2)"]

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    path = reports_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(path)
    plt.close()
    print(f"    Confusion matrix saved: {path}")
    return path


# Hypothesis Evaluation

HYPOTHESES = {
    "H1_BMI_increases_diabetes_risk": [
        "BMI",
        "BMI_cat_Obese",
        "BMI_cat_Overweight",
        "BMI_cat_Underweight",
    ],
    "H2_poor_health_correlates_diabetes": [
        "GenHlth",
        "Total_Unhealthy_Days",
    ],
    "H3_older_age_increases_diabetes_risk": [
        "Age",
        "Age_cat_Old",
        "Age_cat_Young",
    ],
}


def evaluate_hypotheses(model_name, feature_names, importances):
    results = []
    fi_dict = dict(zip(feature_names, importances))
    sorted_importances = sorted(importances, reverse=True)
    total_features = len(feature_names)

    for hypothesis, features in HYPOTHESES.items():
        for feature in features:
            if feature in fi_dict:
                importance = fi_dict[feature]
                rank = sorted_importances.index(importance) + 1
                results.append({
                    "model": model_name,
                    "hypothesis": hypothesis,
                    "feature": feature,
                    "importance": round(importance, 6),
                    "rank": rank,
                    "total_features": total_features,
                    "supported": rank <= total_features // 2,
                })

    return results
