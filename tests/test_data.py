import pandas as pd
from src.features import build_features

def test_build_features_returns_dataframe():
    df = build_features()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

def test_required_features_exist():
    df = build_features()
    required = [
        'sales_lag_1',
        'sales_lag_7',
        'rolling_mean_7',
        'rolling_std_7'
    ]

    for col in required:
        assert col in df.columns

def test_no_nulls_in_model_features():
    df = build_features()
    model_features = [
        'sales_lag_1',
        'sales_lag_7',
        'rolling_mean_7',
        'rolling_std_7'
    ]

    assert df[model_features].isnull().sum().sum() == 0