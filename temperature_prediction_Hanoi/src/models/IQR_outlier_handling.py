
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class IQRTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer loại bỏ hoặc giới hạn (clip) outliers theo phương pháp IQR.
    Có thể đưa vào Pipeline như một step xử lý numeric features.
    """
    def __init__(self, multiplier=1.5, method='clip'):
        """
        Parameters
        ----------
        multiplier : float
            Hệ số nhân với IQR (thường 1.5 hoặc 3.0)
        method : {'clip', 'remove'}
            'clip'  : cắt giá trị vượt ngưỡng về giới hạn IQR
            'remove': loại bỏ các hàng chứa outlier
        """
        self.multiplier = multiplier
        self.method = method
        self.bounds_ = None
        self.numeric_cols_ = None

    def fit(self, X, y=None):
        """Tính Q1, Q3, IQR và lưu lại ngưỡng cho mỗi cột numeric."""
        X_df = pd.DataFrame(X).copy()
        self.numeric_cols_ = X_df.select_dtypes(include=[np.number]).columns
        Q1 = X_df[self.numeric_cols_].quantile(0.25)
        Q3 = X_df[self.numeric_cols_].quantile(0.75)
        IQR = Q3 - Q1
        self.bounds_ = {
            'lower': Q1 - self.multiplier * IQR,
            'upper': Q3 + self.multiplier * IQR
        }
        return self

    def transform(self, X):
        """Áp dụng cắt (clip) hoặc loại bỏ (remove) outlier theo IQR."""
        X_df = pd.DataFrame(X).copy()

        if self.method == 'clip':
            for col in self.numeric_cols_:
                lower = self.bounds_['lower'][col]
                upper = self.bounds_['upper'][col]
                X_df[col] = X_df[col].clip(lower, upper)

        elif self.method == 'remove':
            mask = ~(
                (X_df[self.numeric_cols_] < self.bounds_['lower']) |
                (X_df[self.numeric_cols_] > self.bounds_['upper'])
            ).any(axis=1)
            X_df = X_df[mask]

        return X_df.values if isinstance(X, np.ndarray) else X_df

    def count_outliers(self, X):
        """Hàm phụ để đếm số lượng outlier trong từng cột (giống như bạn làm thủ công)."""
        X_df = pd.DataFrame(X).copy()
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        Q1 = X_df[numeric_cols].quantile(0.25)
        Q3 = X_df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((X_df[numeric_cols] < (Q1 - self.multiplier * IQR)) |
                    (X_df[numeric_cols] > (Q3 + self.multiplier * IQR))).sum()
        return outliers

