from fastapi import APIRouter
from pydantic import BaseModel
from .model import cluster_securities

router = APIRouter()

class ClusterRequest(BaseModel):
    tickers: list[str]
    start_date: str
    end_date: str
    interval: str
    cluster_range: list[int]

@router.post("/cluster/")
def cluster_stocks(request: ClusterRequest):
    """
    Cluster stocks based on list of tickers.
    Example:

    {"tickers": ["AAPL","MSFT","GOOGL","AMZN","TSLA","NFLX",    "NVDA"], "start_date": "2020-01-01", "end_date": 2025-01-01", "interval": "1wk","cluster_range": [2,3]}
    """
    desc_stats, model, k, score, labels = cluster_securities(
        request.tickers,
        request.start_date,
        request.end_date,
        request.interval,
        request.cluster_range
    )
    return {"best_k": k,
            "silhouette_score": score,
            "clusters": [
            {"ticker": ticker, "label": int(label)}
            for ticker, label in zip(request.tickers, labels)]
            }

