from fastapi import APIRouter
from pydantic import BaseModel
from .model import train_and_eval
from rich import print
from fastapi.encoders import jsonable_encoder
from typing import List, Dict

router = APIRouter()

class ClusterPortfolio(BaseModel):
    tickers: List[str]
    weights: List[float]

class TrainEvalRequest(BaseModel):
    tickers: list[str]
    start_date: str
    end_date: str
    interval: str
    cluster_range: list[int]
    test_start_date: str
    test_end_date: str
    baseline_ticker: str

class EvalResponse(BaseModel):
    optimized_portfolio: Dict[str, ClusterPortfolio] #str key...
    model_metrics: List[dict]
    baseline_metrics: List[dict]
    relative_metrics: List[dict]

@router.post("/train_eval/", response_model=EvalResponse)
def eval_stocks(request: TrainEvalRequest):
    """based on the clustering results, optimizes the portfolio weights and compares the results with a baseline portfolio on a test data and returns its metrics
    
    
    Example:
            {
        "tickers": [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "NFLX",
            "NVDA"
        ],
        "start_date": "2020-01-01",
        "end_date": "2021-01-01",
        "interval": "1wk",
        "cluster_range": [
            2,
            3
        ],
        "test_start_date": "2021-05-01",
        "test_end_date": "2021-12-31",
        "baseline_ticker": "NDAQ"
        }
    """
    optimized_portfolio, model_metrics, baseline_metrics, relative_metrics = train_and_eval(
        request.tickers,
        request.start_date,
        request.end_date,
        request.interval,
        request.cluster_range,
        request.baseline_ticker,
        request.test_start_date,
        request.test_end_date
    )
    optimized_portfolio = {str(k): v for k, v in optimized_portfolio.items()}
    print(optimized_portfolio, "OPT")

    def flatten_to_json(metrics_df):
        #converts a wide dataframe into a json parseable format
        out= metrics_df.reset_index().to_dict(orient="records")
        return out

    model_metrics = flatten_to_json(model_metrics)
    #print(model_metrics)
    baseline_metrics = flatten_to_json(baseline_metrics)
    relative_metrics = flatten_to_json(relative_metrics)


    print(relative_metrics, "FINAL")
    #https://stackoverflow.com/questions/65504438/how-to-add-both-file-and-json-body-in-a-fastapi-post-request/70640522#70640522
    # use keyword args in EvalResponse
    return EvalResponse(optimized_portfolio=optimized_portfolio, model_metrics=model_metrics, baseline_metrics=baseline_metrics, relative_metrics=relative_metrics)