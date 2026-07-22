from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200

    data = response.json()

    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True
    assert data['explainer_loaded'] is True

def test_predict_success():
    payload = {
        'Store': 1,
        'DayOfWeek': 2,
        'Promo': 1,
        'SchoolHoliday': 0,
        'CompetitionDistance': 1270,
        'year': 2015,
        'month': 7,
        'day': 15,
        'weekofyear': 29,
        'is_weekend': 0,
        'sales_lag_1': 6200,
        'sales_lag_7': 5900,
        'rolling_mean_7': 6050,
        'rolling_std_7': 420
    }
    response = client.post('/predict', json=payload)

    assert response.status_code == 200

    data = response.json()

    assert 'forecast' in data
    assert isinstance(data['forecast'], (int, float))
    assert len(data['top_contributors']) > 0

def test_predict_invalid_day_of_week():
    payload = {
        'Store': 1,
        'DayOfWeek': 9,
        'Promo': 1,
        'SchoolHoliday': 0,
        'CompetitionDistance': 1270,
        'year': 2015,
        'month': 7,
        'day': 15,
        'weekofyear': 29,
        'is_weekend': 0,
        'sales_lag_1': 6200,
        'sales_lag_7': 5900,
        'rolling_mean_7': 6050,
        'rolling_std_7': 420
    }
    response = client.post('/predict', json=payload)

    assert response.status_code == 422

def test_predict_negative_competition_distance():
    payload = {
        'Store': 1,
        'DayOfWeek': 2,
        'Promo': 1,
        'SchoolHoliday': 0,
        'CompetitionDistance': -100,
        'year': 2015,
        'month': 7,
        'day': 15,
        'weekofyear': 29,
        'is_weekend': 0,
        'sales_lag_1': 6200,
        'sales_lag_7': 5900,
        'rolling_mean_7': 6050,
        'rolling_std_7': 420
    }
    response = client.post('/predict', json=payload)

    assert response.status_code == 422