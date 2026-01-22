"""
model.py
--------
Model building, evaluation, and comparison utilities.
Designed so the main notebook stays readable — implementation lives here.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    classification_report, confusion_matrix, roc_curve, f1_score
)

RANDOM_STATE = 42


def build_models() -> dict:
    """Return a dict of named sklearn Pipelines — one per model family."""
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE
            ))
        ]),
        'Random Forest': Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_leaf=5,
                class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        'Gradient Boosting': Pipeline([
            ('clf', GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=RANDOM_STATE
            ))
        ]),
    }


def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test,
                       cv_splits: int = 5) -> dict:
    """
    Train all models, run cross-validation, and return a results dict.
    Each entry contains the fitted pipeline plus all key metrics.
    """
    results = {}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

        results[name] = {
            'pipeline': pipeline,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'avg_precision': average_precision_score(y_test, y_prob),
            'cv_auc_mean': cv_auc.mean(),
            'cv_auc_std': cv_auc.std(),
        }
        print(
            f"{name:25s}  AUC: {results[name]['roc_auc']:.4f}  |"
            f"  CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}"
        )

    return results


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the threshold that maximises F1 score on the positive class."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    return thresholds[np.argmax(f1_scores)]


def business_cost_analysis(y_true, y_pred, monthly_incomes: np.ndarray,
                            replacement_multiplier: float = 1.5,
                            intervention_cost: float = 500,
                            success_rate: float = 0.30) -> dict:
    """
    Translate model predictions into business impact estimates.
    Returns a dict of cost/savings components.
    """
    annual = monthly_incomes * 12
    tp = (y_pred == 1) & (y_true == 1)
    fp = (y_pred == 1) & (y_true == 0)
    fn = (y_pred == 0) & (y_true == 1)

    savings = (annual[tp] * replacement_multiplier * success_rate).sum()
    interventions = (tp.sum() + fp.sum()) * intervention_cost
    missed = (annual[fn] * replacement_multiplier).sum()

    return {
        'tp': int(tp.sum()), 'fp': int(fp.sum()), 'fn': int(fn.sum()),
        'savings': savings, 'intervention_cost': interventions,
        'missed_cost': missed, 'net_benefit': savings - interventions
    }


def precision_recall_summary(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Quick summary of precision, recall, and F1 at a given threshold."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    preds = (y_prob >= threshold).astype(int)
    return {
        'threshold': threshold,
        'precision': round(precision_score(y_true, preds, zero_division=0), 4),
        'recall': round(recall_score(y_true, preds, zero_division=0), 4),
        'f1': round(f1_score(y_true, preds, zero_division=0), 4),
    }
