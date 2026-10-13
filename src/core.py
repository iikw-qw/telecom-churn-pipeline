import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC


def make_pipeline(estimator, scale=True):
    prep = ColumnTransformer(
        [
            ("num", StandardScaler() if scale else "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("prep", prep), ("clf", estimator)])


def candidates(seed=0):
    return {
        "logistic": make_pipeline(LogisticRegression(max_iter=1000)),
        "random_forest": make_pipeline(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=seed), scale=False
        ),
        "gradient_boosting": make_pipeline(
            HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed), scale=False
        ),
    }


def cross_validate_all(x, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows = {}
    for name, pipe in candidates(seed).items():
        scores = cross_val_score(pipe, x, y, cv=cv, scoring="roc_auc")
        rows[name] = {"auc_mean": scores.mean(), "auc_std": scores.std()}
    return pd.DataFrame(rows).T.sort_values("auc_mean", ascending=False)


def split(df, seed=0):
    x, y = df.drop(columns="churn"), df["churn"]
    return train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)


def campaign_profit_curve(y, p, retained_value=300.0, offer_cost=40.0, save_rate=0.3):
    """Expected profit when contacting the top share of customers by churn probability.

    A contacted churner is saved with probability save_rate and is then worth retained_value;
    every contact costs offer_cost.
    """
    order = np.argsort(-np.asarray(p))
    y_sorted = np.asarray(y)[order]
    n = len(y_sorted)
    contacted = np.arange(1, n + 1)
    churners = np.cumsum(y_sorted)
    profit = churners * save_rate * retained_value - contacted * offer_cost
    share = contacted / n
    return share, profit


def best_campaign_size(y, p, **kw):
    share, profit = campaign_profit_curve(y, p, **kw)
    i = int(np.argmax(profit))
    return float(share[i]), float(profit[i])


def feature_importance(pipe, x, y, seed=0, repeats=5):
    res = permutation_importance(pipe, x, y, scoring="roc_auc", n_repeats=repeats, random_state=seed)
    return pd.Series(res.importances_mean, index=x.columns).sort_values(ascending=False)


def save_model(pipe, path):
    joblib.dump(pipe, path)


def predict_risk(path, customers):
    pipe = joblib.load(path)
    return pipe.predict_proba(customers)[:, 1]
