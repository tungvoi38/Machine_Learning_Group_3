import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

from .IQR_outlier_handling import IQRTransformer


class IQRWrapper(BaseEstimator, TransformerMixin):
    """Wrapper that applies the IQRTransformer only to numeric columns and
    returns a pandas DataFrame (keeps column names). This makes it easy to
    run an IQR-based cleaning step as the first pipeline stage while letting
    subsequent ColumnTransformer steps select by column name.
    """

    def __init__(self, numeric_cols: Optional[List[str]] = None, multiplier: float = 1.5, method: str = 'clip'):
        self.numeric_cols = numeric_cols
        self.multiplier = multiplier
        self.method = method
        self._iqr = None

    def fit(self, X: pd.DataFrame, y=None):
        df = pd.DataFrame(X).copy()
        if self.numeric_cols is None:
            self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._iqr = IQRTransformer(multiplier=self.multiplier, method=self.method)
        # fit only on numeric subset
        self._iqr.fit(df[self.numeric_cols])
        return self

    def transform(self, X: pd.DataFrame):
        df = pd.DataFrame(X).copy()
        if not self.numeric_cols:
            return df
        transformed = self._iqr.transform(df[self.numeric_cols])
        # transform returns DataFrame or ndarray; make into DataFrame matching index/columns
        transformed_df = pd.DataFrame(transformed, index=df.index, columns=self.numeric_cols)
        df.loc[:, self.numeric_cols] = transformed_df
        return df


def compute_outlier_ratios(df: pd.DataFrame, numeric_cols: List[str], multiplier: float = 1.5) -> pd.Series:
    """Return fraction of outliers per numeric column using IQR definition."""
    iqr = IQRTransformer(multiplier=multiplier)
    counts = iqr.count_outliers(df[numeric_cols])
    ratios = counts / float(len(df))
    return ratios


def build_pipeline(
    df_sample: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    nominal_cols: Optional[List[str]] = None,
    ordinal_cols: Optional[List[str]] = None,
    outlier_threshold: float = 0.05,
    iqr_multiplier: float = 1.5,
    iqr_method: str = 'clip',
    select_k: Optional[int] = None,
    model=None,
) -> Pipeline:
    """Build a scikit-learn Pipeline according to these rules:

    - First step: handle outliers (IQR clip/remove) on numeric columns.
    - ColumnTransformer: for numeric columns with many outliers -> RobustScaler,
      numeric with few outliers -> StandardScaler, nominal -> OneHotEncoder,
      ordinal -> OrdinalEncoder.
    - SelectKBest (f_regression) to reduce features (k computed if None).
    - Estimator (RandomForestRegressor by default).

    Parameters
    ----------
    df_sample : pd.DataFrame
        A sample DataFrame used to infer column lists when not provided.
    numeric_cols, nominal_cols, ordinal_cols : lists or None
        Column lists (if None they will be inferred from dtypes; ensure to
        exclude the target column before calling).
    outlier_threshold : float
        Fraction threshold above which a numeric column is considered to have
        "many" outliers (default 0.05 = 5%).
    select_k : int or None
        Number of features to keep in SelectKBest. If None an heuristic is used.
    model : estimator or None
        Final estimator. If None uses RandomForestRegressor(random_state=42).
    """

    df = df_sample.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if nominal_cols is None:
        nominal_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if ordinal_cols is None:
        ordinal_cols = []

    # remove datetime or target if present in the provided lists – caller should
    # pass feature-only columns but we guard here just in case
    for bad in ['datetime', 'temp']:
        if bad in numeric_cols:
            numeric_cols.remove(bad)
        if bad in nominal_cols:
            nominal_cols.remove(bad)
        if bad in ordinal_cols:
            ordinal_cols.remove(bad)

    # determine outlier ratios and split numeric columns
    if numeric_cols:
        ratios = compute_outlier_ratios(df, numeric_cols, multiplier=iqr_multiplier)
        many_outliers = ratios[ratios > outlier_threshold].index.tolist()
        few_outliers = [c for c in numeric_cols if c not in many_outliers]
    else:
        many_outliers = []
        few_outliers = []

    # Imputer + scaler pipelines
    transformers = []
    if many_outliers:
        transformers.append(
            ('num_robust', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), many_outliers)
        )
    if few_outliers:
        transformers.append(
            ('num_std', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), few_outliers)
        )

    if nominal_cols:
        transformers.append(
            ('onehot', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ]), nominal_cols)
        )

    if ordinal_cols:
        transformers.append(
            ('ord', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ord', OrdinalEncoder())
            ]), ordinal_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    # heuristic for select_k when not provided
    n_features_approx = len(numeric_cols) + len(nominal_cols) + len(ordinal_cols)
    if select_k is None:
        # keep up to half of original features but not more than 20
        select_k = max(1, min(20, max(1, n_features_approx // 2)))

    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline([
        ('iqr', IQRWrapper(numeric_cols=numeric_cols, multiplier=iqr_multiplier, method=iqr_method)),
        ('preproc', preprocessor),
        ('select', SelectKBest(score_func=f_regression, k=select_k)),
        ('model', model)
    ])

    return pipeline
