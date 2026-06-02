"""
Find optimal threshold for each model separately.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
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

# Models to evaluate
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
        random_state=42, n_jobs=1, verbosity=0
    ),
}

# Test thresholds
thresholds = np.arange(0.1, 0.91, 0.05)
results = {}

for model_name, model in models.items():
    print(f"\nTesting {model_name}...")
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    precisions = []
    recalls = []
    f1s = []
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f = f1_score(y_test, y_pred, zero_division=0)
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    
    results[model_name] = {
        "thresholds": thresholds,
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
    }
    
    # Find best threshold (max F1)
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    
    print(f"  Best threshold (max F1): {best_threshold:.2f}")
    print(f"  F1: {best_f1:.4f}, Precision: {precisions[best_idx]:.4f}, Recall: {recalls[best_idx]:.4f}")

# Plot both models
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for idx, (model_name, data) in enumerate(results.items()):
    ax = axes[idx]
    ax.plot(data["thresholds"], data["precisions"], marker='o', label='Precision', linewidth=2)
    ax.plot(data["thresholds"], data["recalls"], marker='s', label='Recall', linewidth=2)
    ax.plot(data["thresholds"], data["f1s"], marker='^', label='F1', linewidth=2)
    
    best_idx = np.argmax(data["f1s"])
    best_thresh = data["thresholds"][best_idx]
    ax.axvline(best_thresh, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {best_thresh:.2f}')
    
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'{model_name} - Threshold Optimization', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0, 1])

plt.tight_layout()
output_path = Path(__file__).parent.parent / "reports" / "threshold_optimization.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Threshold optimization plot saved: {output_path}")

# Summary
print("\n" + "="*60)
print("THRESHOLD RECOMMENDATIONS:")
print("="*60)
for model_name, data in results.items():
    best_idx = np.argmax(data["f1s"])
    best_threshold = data["thresholds"][best_idx]
    print(f"\n{model_name}:")
    print(f"  Optimal threshold (max F1): {best_threshold:.2f}")
    print(f"  F1: {data['f1s'][best_idx]:.4f}")
    print(f"  Precision: {data['precisions'][best_idx]:.4f}")
    print(f"  Recall: {data['recalls'][best_idx]:.4f}")
