import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

API_URL = os.getenv('API_URL', 'http://localhost:8000/predict')

st.set_page_config(
    page_title='Zimbabwe Retail Forecast',
    page_icon='📈',
    layout='wide'
)

st.title('📈 Zimbabwe Retail Demand Forecasting')
st.caption('XGBoost + SHAP + FastAPI')

# Sidebar inputs
st.sidebar.header('Forecast Inputs')
store = st.sidebar.selectbox('Store', [1, 2, 3, 4, 5], index=0)
day_of_week = st.sidebar.selectbox(
    'Day of Week',
    options=[1, 2, 3, 4, 5, 6, 7],
    format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x-1]
)
promo = st.sidebar.toggle('Promotion Active', value=True)
school_holiday = st.sidebar.toggle('School Holiday', value=False)
competition_distance = st.sidebar.slider(
    'Competition Distance (m)',
    min_value=100,
    max_value=20000,
    value=1270,
    step=100
)

st.sidebar.subheader('Recent Sales Trend')
sales_lag_1 = st.sidebar.slider(
    'Yesterday Sales',
    min_value=1000,
    max_value=15000,
    value=6200,
    step=100
)
sales_lag_7 = st.sidebar.slider(
    'Sales 7 Days Ago',
    min_value=1000,
    max_value=15000,
    value=5900,
    step=100
)
rolling_mean_7 = st.sidebar.slider(
    '7-Day Average',
    min_value=1000,
    max_value=15000,
    value=6050,
    step=100
)
rolling_std_7 = st.sidebar.slider(
    '7-Day Volatility',
    min_value=50,
    max_value=2000,
    value=420,
    step=10
)

# Build payload
payload = {
    'Store': store,
    'DayOfWeek': day_of_week,
    'Promo': int(promo),
    'SchoolHoliday': int(school_holiday),
    'CompetitionDistance': float(competition_distance),
    'year': 2015,
    'month': 7,
    'day': 15,
    'weekofyear': 29,
    'is_weekend': int(day_of_week >= 6),
    'sales_lag_1': float(sales_lag_1),
    'sales_lag_7': float(sales_lag_7),
    'rolling_mean_7': float(rolling_mean_7),
    'rolling_std_7': float(rolling_std_7)
}

# Predict button
if st.button('🚀 Generate Forecast', type='primary'):
    with st.spinner('Calling forecasting API...'):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Forecast metric
                st.metric(
                    label='Predicted Daily Sales',
                    value=f"{result['forecast']:,.0f} units"
                )
                
                # Contributors
                contrib_df = pd.DataFrame(result['top_contributors'])
                
                # Color by positive/negative impact
                contrib_df['impact'] = contrib_df['shap_value'].apply(
                    lambda x: 'Positive' if x >= 0 else 'Negative'
                )
                
                fig = px.bar(
                    contrib_df,
                    x='shap_value',
                    y='feature',
                    orientation='h',
                    color='impact',
                    title='Top SHAP Drivers',
                    labels={
                        'shap_value': 'Contribution to Forecast',
                        'feature': 'Feature'
                    }
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Raw JSON
                with st.expander('View Raw API Response'):
                    st.json(result)
                    
            else:
                st.error(f'API Error {response.status_code}')
                st.code(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error('Cannot connect to FastAPI backend.')
            # Replaced backticks with repr() for Python 3 compliance
            st.info('Make sure ' + repr('uvicorn api.main:app --reload') + ' is running.')
            
        except Exception as e:
            st.error(f'Unexpected error: {str(e)}')

# Architecture section
with st.expander('Architecture'):
    st.markdown('''
**Frontend:** Streamlit

**Backend:** FastAPI

**Model:** XGBoost

**Explainability:** SHAP

**Experiment Tracking:** MLflow

**Deployment Target:** Docker + Render
    ''')