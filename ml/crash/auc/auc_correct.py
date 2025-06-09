def calculate_auc(y_true, y_scores):
    """
    Calculate AUC-ROC from scratch, properly handling ties in predicted scores.
    Follows the Mann-Whitney U statistic approach.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted scores/probabilities for class 1
        
    Returns:
        auc: Area Under the ROC Curve (AUC-ROC)
    """
    # Combine true labels with predicted scores and sort by score (descending)
    data = sorted(zip(y_scores, y_true), key=lambda x: x[0])
    
    # Count positive and negative samples
    num_pos = sum(1 for y in y_true if y == 1)
    num_neg = len(y_true) - num_pos
    
    if num_pos == 0 or num_neg == 0:
        raise ValueError("Need both positive and negative examples to compute AUC")
    
    # Calculate ranks (handling ties by assigning average rank)
    ranks = []
    i = 0
    n = len(data)
    while i < n:
        j = i + 1
        # Find all elements with the same score
        while j < n and data[j][0] == data[i][0]:
            j += 1
        # Assign average rank to tied elements
        avg_rank = (i + j - 1) / 2
        ranks.extend([avg_rank+1] * (j - i))
        i = j
    
    # Sum ranks for positive class
    rank_sum = sum(rank for (score, label), rank in zip(data, ranks) if label == 1)
    
    # Calculate AUC using Mann-Whitney U formula
    auc = (rank_sum - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg)
    
    return auc

# Example usage
if __name__ == "__main__":
    # Test case with ties
    y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    y_scores = [0.1, 0.3, 0.35, 0.4, 0.4, 0.5, 0.5, 0.7, 0.8, 0.9]  # Contains ties at 0.4 and 0.5
    
    auc = calculate_auc(y_true, y_scores)
    
    print(f"AUC: {auc:.4f}")

    from sklearn.metrics import roc_auc_score
    print(f"Sklearn AUC: {roc_auc_score(y_true, y_scores):.4f}")