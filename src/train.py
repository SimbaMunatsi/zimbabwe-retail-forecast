import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from src.config import DATA_PROCESSED, MODELS_DIR

# ----------------------------
# MLflow Setup
# ----------------------------
mlflow.set_experiment('zimbabwe-retail-forecast')

# ----------------------------
# Load data
# ----------------------------
df = pd.read_parquet(DATA_PROCESSED / 'retail_features.parquet')

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

X = df[FEATURES].fillna(0)
y = df[TARGET]

# ----------------------------
# Time-based split
# ----------------------------
split_date = '2015-06-01'

train_mask = df['Date'] < split_date
test_mask = df['Date'] >= split_date

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f'Train samples: {len(X_train):,}')
print(f'Test samples: {len(X_test):,}')

# ----------------------------
# Model parameters
# ----------------------------
params = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1
}

# ----------------------------
# MLflow Run
# ----------------------------
with mlflow.start_run():

    # Log parameters
    mlflow.log_params(params)

    # Train model
    model = XGBRegressor(**params)

    print('Training XGBoost...')
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, preds)

    print(f'XGBoost MAE: {mae:,.2f}')

    # Log metric
    mlflow.log_metric('mae', mae)

    # Log model to MLflow
    mlflow.xgboost.log_model(model, artifact_path='model')

    # Save local model
    model_path = MODELS_DIR / 'xgboost_retail_model.pkl'
    joblib.dump(model, model_path)

    print(f'Model saved to: {model_path}')

    # Log artifact
    mlflow.log_artifact(model_path)

    # Feature list artifact
    feature_file = MODELS_DIR / 'features.txt'
    with open(feature_file, 'w') as f:
        f.write('\\n'.join(FEATURES))

    mlflow.log_artifact(feature_file)

    print('\\nRun logged successfully to MLflow.')