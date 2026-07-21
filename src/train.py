import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from config import DATA_PROCESSED, MODELS_DIR
from pathlib import Path

# Load processed data
df = pd.read_parquet(DATA_PROCESSED / 'retail_features.parquet')

# Features and target
TARGET = 'Sales'

FEATURES = [
    'Store',
    'DayOfWeek',
    'Promo',
    'SchoolHoliday',
    'CompetitionDistance',
    'year',
    'month',
    'day',
    'weekofyear',
    'is_weekend',
    'sales_lag_1',
    'sales_lag_7',
    'rolling_mean_7',
    'rolling_std_7'
]

# Fill remaining missing values
X = df[FEATURES].fillna(0)
y = df[TARGET]

# Time-based split (NO random split)
split_date = '2015-06-01'

train_mask = df['Date'] < split_date
test_mask = df['Date'] >= split_date

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f'Train samples: {len(X_train):,}')
print(f'Test samples: {len(X_test):,}')

# Basic XGBoost model
model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

# Train
print('Training XGBoost...')
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, preds)

print(f'XGBoost MAE: {mae:,.2f}')

# Save model
model_path = MODELS_DIR / 'xgboost_retail_model.pkl'
joblib.dump(model, model_path)

print(f'Model saved to: {model_path}')


metrics_path = MODELS_DIR / 'metrics.txt'

with open(metrics_path, 'w') as f:
    f.write(f'MAE: {mae:.2f}\n')

print(f'Metrics saved to: {metrics_path}')