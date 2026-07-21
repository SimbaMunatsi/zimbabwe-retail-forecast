import pandas as pd
from sklearn.metrics import mean_absolute_error
from config import DATA_PROCESSED

# Load processed data
df = pd.read_parquet(DATA_PROCESSED / 'retail_features.parquet')

# Same split as XGBoost
split_date = '2015-06-01'

train_mask = df['Date'] < split_date
test_mask = df['Date'] >= split_date

test_df = df[test_mask].copy()

# Naive forecast: use sales from 7 days ago
test_df['naive_pred'] = test_df['sales_lag_7']

# Evaluate
mae = mean_absolute_error(test_df['Sales'], test_df['naive_pred'])

print(f'Naive Baseline MAE: {mae:,.2f}')