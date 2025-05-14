def check_relative_performance(relative_metrics, threshold=1.0):
    """
    Checks if the model's performance is higher than the baseline.
    Input example:
    [
    {
        'cluster': 0,
        'cumulative_return': 0.5138449116553205,
        'annual_return': 0.6341747384613529,
        'annual_volatility': 0.2039787355821499,
        'sharpe_ratio': 3.1090237747157072
    },
    ....
]
    Returns list[tuple[bool, str]] returns which cluster and metric underperformed 
    """


    keys_to_check = ["cumulative_return", "annual_return", "sharpe_ratio"]
    messages = []
    for cluster_metrics in relative_metrics:
        cluster_id = cluster_metrics["cluster"]
        for key in keys_to_check:
            value = cluster_metrics.get(key)
            if value is None or value < threshold:
                messages.append((False, f"{key} for cluster {cluster_id} underperformed (value={value:.2f})"))
    messages.append((True, "All clusters meet performance threshold."))
    return messages