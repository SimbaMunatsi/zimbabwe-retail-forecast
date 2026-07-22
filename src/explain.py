import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from config import DATA_PROCESSED, MODELS_DIR

# ----------------------------

# Load processed dataset

# ----------------------------

print('Loading processed dataset...')

df = pd.read_parquet(DATA_PROCESSED / 'retail_features.parquet')

# ----------------------------

# Feature list (must match training)

# ----------------------------

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

# Create feature matrix

X = df[FEATURES].fillna(0)

print(f'Dataset shape: {X.shape}')

# ----------------------------

# Load trained XGBoost model

# ----------------------------

model_path = MODELS_DIR / 'xgboost_retail_model.pkl'

print(f'Loading model from: {model_path}')

model = joblib.load(model_path)

# ----------------------------

# Create SHAP samples

# ----------------------------

# Small background sample for the explainer

X_background = X.sample(100, random_state=42)

# Larger sample for explanations and plots

X_sample = X.sample(1000, random_state=42)

print(f'Background shape: {X_background.shape}')
print(f'SHAP sample shape: {X_sample.shape}')

# ----------------------------

# Create SHAP explainer

# ----------------------------

# ----------------------------
# Create SHAP explainer (WORKING VERSION)
# ----------------------------

print('Creating SHAP explainer...')

# Model-agnostic explainer compatible with XGBoost 2.1 + Python 3.12
explainer = shap.Explainer(model.predict, X_background)

# Save explainer for API reuse
explainer_path = MODELS_DIR / 'shap_explainer.pkl'
joblib.dump(explainer, explainer_path)

print(f'SHAP explainer saved to: {explainer_path}')

# ----------------------------
# Calculate SHAP values
# ----------------------------

print('Calculating SHAP values...')

# Returns a SHAP Explanation object
shap_explanation = explainer(X_sample)

# Extract raw SHAP values
shap_values_array = shap_explanation.values

print('SHAP values calculated successfully.')

# ----------------------------

# Save SHAP summary plot

# ----------------------------

print('Generating SHAP summary plot...')

plt.figure(figsize=(12, 8))

shap.summary_plot(
shap_values_array,
X_sample,
show=False
)

summary_path = MODELS_DIR / 'shap_summary.png'

plt.savefig(
summary_path,
bbox_inches='tight',
dpi=300
)

plt.close()

print(f'SHAP summary plot saved to: {summary_path}')

# ----------------------------

# Calculate feature importance

# ----------------------------

importance = pd.DataFrame({
'feature': FEATURES,
'importance': abs(shap_values_array).mean(axis=0)
}).sort_values('importance', ascending=False)

# Save importance table

importance_path = MODELS_DIR / 'shap_feature_importance.csv'
importance.to_csv(importance_path, index=False)

print(f'Feature importance saved to: {importance_path}')

# ----------------------------

# Display top features

# ----------------------------

print('\n' + '=' * 50)
print('TOP 10 SHAP FEATURES')
print('=' * 50)

print(importance.head(10).to_string(index=False))

# ----------------------------

# Example explanation for one prediction

# ----------------------------

print('\n' + '=' * 50)
print('EXAMPLE PREDICTION EXPLANATION')
print('=' * 50)

sample_idx = 0

sample_shap = shap_values_array[sample_idx]

# Predicted value

prediction = model.predict(X_sample.iloc[[sample_idx]])[0]

print(f'Predicted sales: {prediction:,.2f}')

# Top contributors

contrib_df = pd.DataFrame({
'feature': FEATURES,
'shap_value': sample_shap
})

contrib_df['abs_shap'] = contrib_df['shap_value'].abs()
contrib_df = contrib_df.sort_values('abs_shap', ascending=False)

print('\nTop positive/negative contributors:')

for _, row in contrib_df.head(5).iterrows():
    sign = '+' if row['shap_value'] >= 0 else '-'
    print(f'{row["feature"]:20s} {sign}{abs(row["shap_value"]):,.2f}')

print('\nSHAP analysis completed successfully!')
