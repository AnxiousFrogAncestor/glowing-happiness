from .finance_api import get_prices
import pandas as pd
import numpy as np
from .model_eval import evaluate_cluster_portfolios
from rich import print
import functools
from functools import cached_property
import time
import dask

#get_prices("AAPL", start_date="2020-01-01", end_date="2021-01-01")

#data = get_prices(["AAPL", "MSFT"], start_date="2020-01-01", end_date="2021-01-01")

#https://arxiv.org/pdf/2501.12074

def compute_log_returns(ticker:str, start_date:str, end_date:str, interval:str):
    """Returns the log returns on the adjusted prices.

    Args:
        ticker (str|list[str]): ticker symbol
        start_date (str): start date of stock price
        end_date (str): end date of stock price (e.g. 2025-01-01)
        interval (str): time-granularity of the data e.g. 1wk, 1d

    Returns:
        pd.DataFrame: log returns for each ticker in the list
    """
    data = get_prices(ticker, start_date, end_date, interval)

    #data = data.reset_index()

    #print(data.columns)
    # get the values at the ticker label
    #print(data[ticker])
    data = pd.DataFrame(data[ticker])
    log_returns = np.log(data/data.shift(1))
    log_returns = log_returns.dropna()
    #print(log_returns, type(log_returns))
    return log_returns.dropna()


#compute_log_returns(["AAPL", "MSFT"], start_date="2020-01-01", end_date="2021-01-01", interval="1wk")

import dask
import dask.dataframe as dd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dask import delayed
from dask.distributed import Client

def fit_kmeans_and_evaluate(desc_stats_pd: pd.DataFrame, k: int):
    """
    Fits a KMeans model for a given k on the descriptive-stats DataFrame,
    makes predictions, and evaluates it with the silhouette score.

    Args:
        desc_stats_pd (pd.DataFrame): DataFrame of descriptive stats (mean, std) per ticker.
            Shape: (n_tickers, 2)
        k (int): Number of clusters for KMeans.

    Returns:
        score (float): Silhouette score for this k.
        labels (np.ndarray): Predicted cluster labels for each ticker.
        k (int): The number of clusters used.
        model (KMeans): The fitted KMeans model.
    """
    model = KMeans(n_clusters=k, random_state=42)
    X = desc_stats_pd.to_numpy() # shape (n_tickers, 2)
    model.fit(X)
    labels = model.predict(X)
    score = silhouette_score(X, labels)
    return score, labels, k, model

def clustering_model_with_parallelization(log_returns_pd: pd.DataFrame, number_of_clusters_range: list):
    """
    Perform KMeans clustering with parallel hyperparameter optimization using silhouette score,
    operating on descriptive statistics of log returns.

    Args:
        log_returns_pd (pd.DataFrame): Adjusted price log returns (time × tickers).
        number_of_clusters_range (list[int]): List of cluster counts to evaluate.

    Returns:
        desc_stats_pd (pd.DataFrame): Descriptive stats (mean, std) per ticker.
        best_model (KMeans): The best-fitted sklearn KMeans model.
        best_k (int): The optimal number of clusters.
        best_score (float): Silhouette score of the best model.
        best_labels (np.ndarray): Cluster labels from the best model.
    """

    means = log_returns_pd.mean(axis=0)
    stds  = log_returns_pd.std(axis=0)
    desc_stats_pd = pd.DataFrame({'mean': means, 'std': stds})

    tasks = [
        delayed(fit_kmeans_and_evaluate)(desc_stats_pd, k)
        for k in number_of_clusters_range
    ]
    results = dask.compute(*tasks)

    # select best number of cluster by max silhouette score
    best_score = -1
    best_model = None
    best_k     = None
    best_labels= None

    for score, labels, k, model in results:
        if score > best_score:
            best_score  = score
            best_k      = k
            best_model  = model
            best_labels = labels

    return desc_stats_pd, best_model, best_k, best_score, best_labels

#compute_log_returns(["AAPL", "MSFT"], start_date="2020-01-01", end_date="2021-01-01", interval="1wk")

#TODO cache properly by converting list to tuples to enable hashing
#@functools.lru_cache(maxsize=None)
def cluster_securities(ticker:list, start_date:str, end_date:str, interval:str, cluster_range:list[int]):
    log_returns_df = compute_log_returns(ticker, start_date, end_date, interval)
    desc_stats_pd, best_model, best_k, best_score, best_labels = clustering_model_with_parallelization(log_returns_df, cluster_range)
    return desc_stats_pd, best_model, best_k, best_score, best_labels

import pandas as pd
import numpy as np

def partition_log_returns_by_cluster(log_returns_df: pd.DataFrame, cluster_labels: np.ndarray) -> dict:
    """
    Partitions the columns (tickers) of log_returns_df into separate DataFrames by their cluster label.

    Args:
        log_returns_df (pd.DataFrame): DataFrame of shape (n_time, n_tickers) with log-returns.
        cluster_labels (np.ndarray): 1D array of shape (n_tickers,) giving cluster assignments, 
                                     assumed to correspond in order to log_returns_df.columns.

    Returns:
        dict[int, pd.DataFrame]: A dictionary mapping cluster label to DataFrame with columns from that cluster.
    """
    if len(cluster_labels) != log_returns_df.shape[1]:
        raise ValueError("Length of cluster_labels must match number of columns in log_returns_df")

    cluster_map = {}
    for cluster_id in np.unique(cluster_labels):
        cluster_index = np.array(cluster_labels == cluster_id)
        tickers_in_cluster = log_returns_df.columns[cluster_index]
        cluster_map[cluster_id] = log_returns_df[tickers_in_cluster]

    return cluster_map

import numpy as np
from scipy.optimize import minimize

def portfolio_optimization(mean_returns, cov_matrix, risk_free_rate=0):
    """
    Performs portfolio optimization to maximize the Sharpe Ratio.

    Args:
        mean_returns (np.ndarray): Expected returns for each asset (1D array).
        Shape(num_assets, )
        cov_matrix (np.ndarray): Covariance matrix of asset returns (2D array).
        Shape(num_assets, num_assets)
        risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.

    Returns:
        np.ndarray: Optimal asset weights that maximize the Sharpe Ratio. (#assets, )

    Note: Sharpe ratio formula: $E[R_asset-R_free]/sqrt(var(R_asset-R_free))$, where the numerator is the expected excess return for the asset or portfolio, and the denominator is the standard deviation of the asset/portfolio.
    # https://en.wikipedia.org/wiki/Sharpe_ratio 
    """
    num_assets = len(mean_returns)
    #uniform weighting of the portfolio
    initial_weights = np.ones(num_assets) / num_assets

    # constraint: weights sum to 1, all assets are invested
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    # bounds: weights between 0 and 1 (no short selling, no leverage)
    bounds = tuple((0, 1) for _ in range(num_assets))

    def neg_sharpe_ratio(weights):
        #change to negative for minimization
        portfolio_return = np.dot(weights, mean_returns)
        #normalize by the volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility

    result = minimize(
        neg_sharpe_ratio,
        initial_weights,
        method='SLSQP', #sequential least squares
        bounds=bounds,
        constraints=constraints
    )
    # get optimal portfolio weights (#assets, )
    return result.x

from dask import delayed, compute

def optimize_portfolios_for_clusters(clustered_log_returns, risk_free_rate=0.0):
    """
    Runs portfolio optimization in parallel for each cluster.

    Args:
        clustered_log_returns (dict[int, pd.DataFrame]): Clustered log returns per cluster ID.
        risk_free_rate (float): Risk-free rate used in Sharpe ratio.

    Returns:
        dict[int, dict]: Mapping from cluster_id to a dictionary with:
            - 'tickers': list of tickers in cluster
            - 'weights': np.ndarray of optimal weights
    """
    tasks = {}
    for cluster_id, df in clustered_log_returns.items():
        task = delayed(_optimize_single_cluster)(cluster_id, df, risk_free_rate)
        tasks[cluster_id] = task

    results = compute(*tasks.values())
    return dict(results)

def _optimize_single_cluster(cluster_id, cluster_df, risk_free_rate):
    tickers = cluster_df.columns.tolist()
    mean_returns = cluster_df.mean().values
    cov_matrix = cluster_df.cov().values
    weights = portfolio_optimization(mean_returns, cov_matrix, risk_free_rate)
    return cluster_id, {'tickers': tickers, 'weights': weights}

"""if __name__ == '__main__':
    sample_tickers = [
    'AAPL','MSFT','GOOGL','AMZN','TSLA','NFLX','NVDA']
    client = Client()
    log_returns_df = compute_log_returns(sample_tickers, start_date="2020-01-01", end_date="2021-01-01", interval="1wk")

    test_log_returns_df = compute_log_returns(sample_tickers, start_date="2021-05-01", end_date="2021-12-31", interval="1wk")

    log_returns_baseline = compute_log_returns("NDAQ", start_date="2021-05-01", end_date="2021-12-31", interval="1wk")

    cluster_range = [3, 4, 5, 6]

    desc_stats_pd, best_model, best_k, best_score, best_labels = clustering_model_with_parallelization(log_returns_df, cluster_range)
    print(best_k, "best_k")

    cluster_dict = partition_log_returns_by_cluster(log_returns_df, best_labels)

    clustered_weights = optimize_portfolios_for_clusters(cluster_dict)
    print(clustered_weights)

    model_metrics, baseline_metrics, relative_metrics = evaluate_cluster_portfolios(test_log_returns_df, clustered_weights, log_returns_baseline)
    print(relative_metrics["cumulative_return"])
    print(relative_metrics["annualized_return"])
    print(relative_metrics["annual_volatility"])
    print(relative_metrics["sharpe_ratio"])"""

def train_and_eval(ticker:list[str], start_date:str, end_date:str, interval:str, cluster_range:list[int], baseline:str, test_start_date:str, test_end_date:str):
    #trains the model and evaluates it, compared to the baseline
    #baseline is the ticker of the baseline portfolio e.g. NASDAQ
    client = Client()
    """Returns:
        Tuple of pd.DataFrames:
            model_metrics: Metrics per cluster
            baseline_metrics: Overall baseline metrics
            relative_metrics: Model-to-baseline ratios for each metric"""
    print(cluster_range, "cluster_range")
    desc_stats_pd, best_model, best_k, best_score, best_labels = cluster_securities(ticker, start_date, end_date, interval, cluster_range)

    test_log_returns_df = compute_log_returns(ticker, start_date=test_start_date, end_date=test_end_date, interval=interval)

    log_returns_baseline = compute_log_returns(baseline, start_date=test_start_date, end_date=test_end_date, interval=interval)

    cluster_dict = partition_log_returns_by_cluster(test_log_returns_df, best_labels)

    clustered_weights = optimize_portfolios_for_clusters(cluster_dict)
    print(clustered_weights)

    def make_json_serializable(data, key):
        #convert the numpy array into list for fastAPI json-friendly
        for val in data.values():
            val[key] = val[key].tolist()
        return data
    
    clustered_weights = make_json_serializable(clustered_weights, "weights")
        

    model_metrics, baseline_metrics, relative_metrics = evaluate_cluster_portfolios(test_log_returns_df, clustered_weights, log_returns_baseline)
    print(relative_metrics["cumulative_return"])
    print(relative_metrics["annualized_return"])
    print(relative_metrics["annual_volatility"])
    print(relative_metrics["sharpe_ratio"])
    return clustered_weights, model_metrics, baseline_metrics, relative_metrics

    
