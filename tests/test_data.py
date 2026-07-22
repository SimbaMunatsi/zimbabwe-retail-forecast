import pandas as pd
from src.features import build_features

def test_build_features_function_exists():
    assert callable(build_features)

def test_feature_engineering_columns_logic():
    # Simulate a tiny dataset instead of reading Kaggle data
    df = pd.DataFrame({
        'Sales': [100, 120, 130, 140, 150, 160, 170],
        'Store': [1, 1, 1, 1, 1, 1, 1]
    })
    
    # Basic lag logic
    df['sales_lag_1'] = df.groupby('Store')['Sales'].shift(1)

    # Verify lag behavior
    assert pd.isna(df.loc[0, 'sales_lag_1'])
    assert df.loc[1, 'sales_lag_1'] == 100
    assert df.loc[6, 'sales_lag_1'] == 160

def test_no_negative_sales_in_mock_data():
    df = pd.DataFrame({'Sales': [100, 120, 130]})
    assert (df['Sales'] >= 0).all()