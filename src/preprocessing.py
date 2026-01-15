"""
preprocessing.py
----------------
Data preparation utilities for the employee attrition project.
Reusable across notebooks and downstream pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TARGET = 'Attrition'
DROP_COLS = ['EmployeeNumber']
RANDOM_STATE = 42


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Attrition rate: {df[TARGET].eq('Yes').mean()*100:.1f}%")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that add signal beyond what's in the raw columns.
    Each feature here was motivated by domain reasoning, not just feature importance scores.
    """
    df = df.copy()
    # Income efficiency — how well-paid is this person relative to their experience?
    df['IncomePerYearExperience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    # What fraction of their total career has been at this company?
    df['TenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    # Composite satisfaction — single score aggregating four survey dimensions
    df['SatisfactionScore'] = (
        df['JobSatisfaction'] + df['EnvironmentSatisfaction'] +
        df['RelationshipSatisfaction'] + df['WorkLifeBalance']
    ) / 4
    # Binary flags for known high-risk conditions
    df['PromotionStagnation'] = (df['YearsSinceLastPromotion'] > 4).astype(int)
    df['LongCommute'] = (df['DistanceFromHome'] > 15).astype(int)
    return df


def encode_and_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    One-hot encode categoricals, encode target, and return train/test splits.
    """
    df = df.drop(columns=DROP_COLS, errors='ignore')
    y = (df[TARGET] == 'Yes').astype(int)
    X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def save_processed(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"Saved → {path}")


def satisfaction_composite(df):
    """Compute a single composite satisfaction score from the four survey dimensions."""
    cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
    existing = [c for c in cols if c in df.columns]
    return df[existing].mean(axis=1)
