import numpy as np
import pandas as pd

def evaluate_cluster_portfolios(
    test_log_returns_df: pd.DataFrame,
    cluster_portfolios: dict,
    baseline_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 52
):
    """
    Evaluates performance metrics for each cluster portfolio and compares them to a baseline.

    Args:
        test_log_returns_df (pd.DataFrame): Log returns for test_data (rows = time, columns = tickers).
        cluster_portfolios (dict): Mapping cluster_id -> {'tickers': [...], 'weights': np.array}.
        baseline_returns (pd.Series): Log returns of the baseline portfolio (same time index) on the test_data.
        risk_free_rate (float): Risk-free rate per period (e.g., weekly). Default is 0.
        periods_per_year (int): Number of return periods per year (e.g., 52 for weekly data).

    Returns:
        Tuple of pd.DataFrames:
            model_metrics: Metrics per cluster
            baseline_metrics: Overall baseline metrics
            relative_metrics: Model-to-baseline ratios for each metric
    Notes: https://www.investopedia.com/terms/a/annualized-total-return.asp
    """
    baseline_returns = baseline_returns.reset_index(drop=True)
    if len(baseline_returns) != len(test_log_returns_df):
        raise ValueError("baseline_returns must be the same length as test_log_returns_df")
    holding_periods = len(baseline_returns)
    model_results = []
    #print(baseline_returns, "BASE")
    for cluster_id, data in cluster_portfolios.items():
        tickers = data['tickers']
        weights = np.array(data['weights'])

        cluster_returns = test_log_returns_df[tickers].dot(weights)

        cumulative_return = np.exp(cluster_returns.sum()) - 1
        annualized_return = (1 + cumulative_return) ** (periods_per_year / holding_periods) - 1
        annual_volatility = cluster_returns.std() * np.sqrt(periods_per_year)
        if annual_volatility != 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / annual_volatility
        else:
            sharpe_ratio = np.nan

        model_results.append({
            'cluster': cluster_id,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio
        })

    model_metrics = pd.DataFrame(model_results).set_index('cluster')

    baseline_cumulative_return = np.exp(baseline_returns.sum()) - 1

    baseline_annualized_return = (1 + baseline_cumulative_return) ** (periods_per_year / holding_periods) - 1

    baseline_annual_volatility = baseline_returns.std() * np.sqrt(periods_per_year)

    baseline_annual_vol = baseline_annual_volatility.iloc[0]

    if baseline_annual_vol != 0:
        baseline_sharpe = (baseline_annualized_return - risk_free_rate) / baseline_annual_volatility
    else:
        baseline_sharpe = np.nan

    baseline_metrics = pd.DataFrame([{
        'cumulative_return': baseline_cumulative_return.item(),
        'annualized_return': baseline_annualized_return.item(),
        'annual_volatility': baseline_annual_volatility.item(),
        'sharpe_ratio': baseline_sharpe.item()
    }])

    baseline_metrics = baseline_metrics.reset_index(drop=True)
    relative_metrics = model_metrics / baseline_metrics.iloc[0]

    return model_metrics, baseline_metrics, relative_metrics
