from pandas import read_csv

def get_best_iteration(path, metrics_names, return_metrics=False):
    # Read iterations as DataFrame
    iterations = read_csv(path)
    
    # Find the maximum value for each metric across all iterations
    max_values = [
        max(iterations.iloc[j][i] for j in range(len(iterations))) 
        for i in metrics_names
        ]
    
    # For each iteration, calculate the "balance score" for metrics
    best_iter = None
    best_balance_score = float('-inf')
    for i in range(len(iterations)):
        metrics = iterations.iloc[i][metrics_names]
        # Calculate the sum of differences from the max values to determine how close the list of metrics is
        differences = [abs(i - v) for i, v in zip(metrics, max_values)]
        balance_score = -sum(differences)  # Negate the sum of differences (higher means better balance)

        if balance_score > best_balance_score:
            best_balance_score = balance_score
            best_iter = metrics if return_metrics else iterations.iloc[i].drop(columns=metrics_names)
    
    return best_iter.to_dict()