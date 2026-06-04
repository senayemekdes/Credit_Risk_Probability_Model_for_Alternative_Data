from pydantic import BaseModel
from typing import Optional


class CustomerInput(BaseModel):
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: int

    Amount: float
    Value: float

    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int

    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: float
    std_transaction_amount: float
    total_value: float


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int