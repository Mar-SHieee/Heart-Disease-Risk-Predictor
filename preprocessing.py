import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class RowWiseFeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["bp_chol_ratio"] = X["trestbps"] / X["chol"]
        X["age_hr_interaction"] = X["age"] * X["thalach"]
        X["chest_pain_ecg"] = X["cp"].astype(str) + "_" + X["restecg"].astype(str)
        return X


class RiskScoreAdder(BaseEstimator, TransformerMixin):
    def __init__(self, risk_features=None):
        self.risk_features = risk_features or ["thal", "ca", "oldpeak", "thalach", "exang", "cp"]

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index, name="target")

        corr = X[self.risk_features].join(y).corr()["target"].drop("target")
        self.weights_ = corr.abs()

        self.feature_scaler_ = MinMaxScaler()
        scaled = pd.DataFrame(
            self.feature_scaler_.fit_transform(X[self.risk_features]),
            columns=self.risk_features,
            index=X.index
        )
        scaled["thalach"] = 1 - scaled["thalach"]

        raw_score = sum(scaled[col] * w for col, w in self.weights_.items())

        self.score_scaler_ = MinMaxScaler()
        self.score_scaler_.fit(raw_score.to_frame())
        return self

    def transform(self, X):
        X = X.copy()
        scaled = pd.DataFrame(
            self.feature_scaler_.transform(X[self.risk_features]),
            columns=self.risk_features,
            index=X.index
        )
        scaled["thalach"] = 1 - scaled["thalach"]

        raw_score = sum(scaled[col] * w for col, w in self.weights_.items())
        X["risk_score"] = self.score_scaler_.transform(raw_score.to_frame())
        return X


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns or [
            "risk_score", "age", "trestbps", "chol", "thalach",
            "oldpeak", "age_hr_interaction", "bp_chol_ratio"
        ]

    def fit(self, X, y=None):
        self.bounds_ = {}
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            X[col] = np.clip(X[col], lower, upper)
        return X
