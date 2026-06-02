"""
Precision-Recall curve for XGBoost model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Load processed data
data_path = Path(__file__).parent.parent / config["data"]["processed_path"]
df = pd.read_csv(data_path)

# Prepare data
target = config["model"]["target_column"]
drop_cols = [target, "patient_id", "event_timestamp"]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
y = df[target].copy()

# Convert to binary
if y.nunique() > 2:
    y = (y >= 1).astype(int)

# Encode categorical
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if categorical_cols:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["model"]["test_size"],
    random_state=config["model"]["random_state"],
    stratify=y,
)

# Train XGBoost
print("Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    objective="binary:logistic",
    random_state=config["model"]["random_state"],
    n_jobs=1,
    verbosity=0,
)
xgb_model.fit(X_train, y_train)

# Get probabilities
y_probs = xgb_model.predict_proba(X_test)[:, 1]

# Compute PR curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)
avg_precision = average_precision_score(y_test, y_probs)

# Plot
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC={pr_auc:.3f})')
ax.axhline(y=y_test.mean(), color='r', linestyle='--', label=f'Baseline (pos rate={y_test.mean():.3f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title(f'Precision-Recall Curve - XGBoost\nAvg Precision = {avg_precision:.3f}', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Save
output_path = Path(__file__).parent.parent / "reports" / "precision_recall_xgboost.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Precision-Recall curve saved: {output_path}")

# Print metrics
print(f"Average Precision: {avg_precision:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
