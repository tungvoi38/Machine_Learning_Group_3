import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.metrics import mean_squared_error, r2_score

from .pipeline import build_pipeline

def train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ordinal_cols: Optional[list] = None,
    select_k: Optional[int] = None,
    outlier_threshold: float = 0.05,
    iqr_multiplier: float = 1.5,
    iqr_method: str = 'clip',
    save_path: str = '../results/models/pipeline_model.joblib',
    model=None,
):
    """
    Build and train pipeline from already split data.

    This function expects the caller already performed the time-based split
    (so it does not do any splitting). It infers numeric/nominal features
    from X_train if not provided.
    """
    # infer column groups
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    nominal_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    if ordinal_cols is None:
        ordinal_cols = []

    # build pipeline using train dataframe as sample (adds iqr wrapper etc.)
    sample_df = pd.concat([X_train, y_train], axis=1)
    pipeline = build_pipeline(
        sample_df,
        numeric_cols=numeric_cols,
        nominal_cols=nominal_cols,
        ordinal_cols=ordinal_cols,
        outlier_threshold=outlier_threshold,
        iqr_multiplier=iqr_multiplier,
        iqr_method=iqr_method,
        select_k=select_k,
        model=model,
    )

    print("Fitting pipeline...")
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"Test RMSE: {rmse:.4f}, R2: {r2:.4f}")
    # Ensure the directory for save_path is a directory. If a file with the
    # same name exists, raise a clear error to avoid the ambiguous OSError
    dirpath = os.path.dirname(save_path) or '.'
    if os.path.exists(dirpath):
        if not os.path.isdir(dirpath):
            raise FileExistsError(
                f"Cannot create directory '{dirpath}' because a file with the same name exists. "
                "Please remove or rename that file, or provide a different `save_path`."
            )
    else:
        os.makedirs(dirpath, exist_ok=True)

    joblib.dump(pipeline, save_path)
    print(f"Saved pipeline to {save_path}")

    return pipeline, {'rmse': rmse, 'r2': r2}

if __name__ == '__main__':
    # convenience entrypoint if someone wants to run script directly
    data_path = '../data/processed/data_daily_after_basic_understand.xlsx'
    try:
        df = pd.read_excel(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        n = len(df)
        train_end = int(n * 0.8)
        train = df.iloc[: max(0, train_end - 7)].copy()
        test = df.iloc[train_end:].copy()
        X_train = train.drop(columns=['temp', 'datetime'])
        y_train = train['temp']
        X_test = test.drop(columns=['temp', 'datetime'])
        y_test = test['temp']
        train_pipeline(X_train, y_train, X_test, y_test)
    except Exception as e:
        print('Failed to run sample training entrypoint:', e)