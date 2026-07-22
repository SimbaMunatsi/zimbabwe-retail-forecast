from pydantic import BaseModel, Field
from typing import List


class ForecastRequest(BaseModel):
    Store: int = Field(..., gt=0)
    DayOfWeek: int = Field(..., ge=1, le=7)
    Promo: int = Field(..., ge=0, le=1)
    SchoolHoliday: int = Field(..., ge=0, le=1)
    CompetitionDistance: float = Field(..., ge=0)

    year: int
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    weekofyear: int = Field(..., ge=1, le=53)
    is_weekend: int = Field(..., ge=0, le=1)

    sales_lag_1: float = Field(..., ge=0)
    sales_lag_7: float = Field(..., ge=0)
    rolling_mean_7: float = Field(..., ge=0)
    rolling_std_7: float = Field(..., ge=0)


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class ForecastResponse(BaseModel):
    forecast: float
    top_contributors: List[FeatureContribution]
    model_version: str