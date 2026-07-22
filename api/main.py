import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import MODELS_DIR
from api.schemas import (
    ForecastRequest,
    ForecastResponse,
    FeatureContribution,
)

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title='Zimbabwe Retail Forecast API',
    description='XGBoost retail demand forecasting with SHAP explanations',
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc',
    openapi_url='/openapi.json'
)

# --------------------------------------------------
# CORS (needed for Streamlit frontend)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# --------------------------------------------------
# Feature order (must match training exactly)
# --------------------------------------------------
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

# --------------------------------------------------
# Global placeholders
# These are populated during application startup.
# Keeping them None allows CI tests to import the app
# even when model artifacts are not present.
# --------------------------------------------------
model = None
explainer = None

# --------------------------------------------------
# Startup event: load model artifacts
# --------------------------------------------------
@app.on_event('startup')
def load_artifacts():
    global model, explainer

    print('Loading model artifacts...')

    model_path = MODELS_DIR / 'xgboost_retail_model.pkl'

    # In GitHub Actions / CI the model file may not exist.
    # We allow the app to start so tests can still run.
    if not model_path.exists():
        print('Model artifact not found. Running in CI/test mode.')
        return

    # Load trained model
    model = joblib.load(model_path)

    # Background sample for SHAP
    background_df = pd.DataFrame([{
        'Store': 1,
        'DayOfWeek': 1,
        'Promo': 0,
        'SchoolHoliday': 0,
        'CompetitionDistance': 1000,
        'year': 2015,
        'month': 7,
        'day': 1,
        'weekofyear': 27,
        'is_weekend': 0,
        'sales_lag_1': 5000,
        'sales_lag_7': 5000,
        'rolling_mean_7': 5000,
        'rolling_std_7': 500
    }])

    # Working SHAP approach for your environment
    explainer = shap.Explainer(model.predict, background_df)

    print('Artifacts loaded successfully.')

# --------------------------------------------------
# Health endpoint
# --------------------------------------------------
@app.get('/health')
def health_check():
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'explainer_loaded': explainer is not None
    }

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post('/predict', response_model=ForecastResponse)
def predict(request: ForecastRequest):

    # If artifacts are missing, return a proper API error
    if model is None or explainer is None:
        raise HTTPException(
            status_code=503,
            detail='Model artifacts are not loaded.'
        )

    # Convert request to DataFrame
    input_df = pd.DataFrame([request.model_dump()])

    # Ensure feature order matches training
    input_df = input_df[FEATURES]

    # ----------------------------
    # Forecast
    # ----------------------------
    prediction = float(model.predict(input_df)[0])

    # ----------------------------
    # SHAP explanation
    # ----------------------------
    shap_explanation = explainer(input_df)
    shap_values = shap_explanation.values[0]

    contrib_df = pd.DataFrame({
        'feature': FEATURES,
        'shap_value': shap_values
    })

    # Top 5 contributors by absolute impact
    contrib_df['abs_shap'] = contrib_df['shap_value'].abs()

    top_contribs = (
        contrib_df
        .sort_values('abs_shap', ascending=False)
        .head(5)
    )

    contributors = [
        FeatureContribution(
            feature=row['feature'],
            shap_value=float(row['shap_value'])
        )
        for _, row in top_contribs.iterrows()
    ]

    return ForecastResponse(
        forecast=round(prediction, 2),
        top_contributors=contributors,
        model_version='xgboost_v1'
    )