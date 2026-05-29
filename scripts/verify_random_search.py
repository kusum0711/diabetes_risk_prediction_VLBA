import yaml
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

# Load config
with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)

n_iter = cfg['model'].get('random_search_n_iter', 20)
param_distributions = cfg['models']['logistic_regression']['param_grid']

# Small synthetic dataset
X, y = make_classification(
    n_samples=200,
    n_features=10,
    n_informative=5,
    n_redundant=0,
    n_classes=3,
    random_state=cfg['model']['random_state'],
)

model = LogisticRegression(random_state=cfg['model']['random_state'], class_weight='balanced', max_iter=500)

rs = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=min(n_iter, 4),
    cv=2,
    scoring='recall_macro',
    random_state=cfg['model']['random_state'],
    n_jobs=-1,
    verbose=1,
)

rs.fit(X, y)
print('Best params:', rs.best_params_)
print('Best score:', rs.best_score_)
