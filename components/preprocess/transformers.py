from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SepsisValidator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Validation logic
        required = ['HR', 'Temp']
        if not all(c in X.columns for c in required):
            raise ValueError("Data missing critical vitals.")
        return X
    
    def get_feature_names_out(self, input_features=None):
        # This tells the next step what the columns are named
        return input_features

class SepsisImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Store medians so we use TRAINING medians for TEST data (avoid leakage)
        self.medians_ = X.median(numeric_only=True)
        return self
    
    def transform(self, X):
        return X.fillna(self.medians_)
    
    def get_feature_names_out(self, input_features=None):
        # This tells the next step what the columns are named
        return input_features
    
class SepsisCleaner(BaseEstimator, TransformerMixin):
    """
    Handles dropping all-NaN columns and filling remaining NaNs.
    """
    def fit(self, X, y=None):
        # Calculate medians during fit so we can use them during transform
        self.medians_ = X.median(numeric_only=True)
        # Identify columns that are 100% NaN to drop them
        self.all_nan_cols_ = X.columns[X.isnull().all()].tolist()
        return self

    def transform(self, X):
        X_copy = X.copy()
        # 1. Drop 100% NaN columns
        X_copy = X_copy.drop(columns=self.all_nan_cols_, errors='ignore')
        # 2. Fill NaNs with medians calculated during fit
        X_copy = X_copy.fillna(self.medians_)
        # 3. Final safety fill for anything missed
        return X_copy.fillna(0)

    def get_feature_names_out(self, input_features=None):
        # This tells the next step what the columns are named
        return input_features

class VarianceSelector(BaseEstimator, TransformerMixin):
    """
    Identifies and drops columns with zero variance (constant columns).
    """
    def fit(self, X, y=None):
        # Identify columns where std dev is 0
        variances = X.var(numeric_only=True)
        self.constant_cols_ = variances[variances == 0].index.tolist()
        return self

    def transform(self, X):
        # Drop the constant columns so StandardScaler doesn't crash
        return X.drop(columns=self.constant_cols_, errors='ignore')
    
    def get_feature_names_out(self, input_features=None):
        # This tells the next step what the columns are named
        return input_features