import pytest
from fastapi.testclient import TestClient
from main import app
from validation import check_relative_performance

client = TestClient(app)

def test_integration():
    payload = {"tickers": [ "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NFLX", "NVDA" ], "start_date": "2020-01-01", "end_date": "2021-01-01", "interval": "1wk", "cluster_range": [ 2, 3 ], "test_start_date": "2021-05-01", "test_end_date": "2021-12-31", "baseline_ticker": "NDAQ" }

    response = client.post("/api/v1/model/train_eval", json=payload)
    assert response.status_code == 200
