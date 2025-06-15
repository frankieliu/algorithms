def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC) from scratch.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or scores for the positive class
        
    Returns:
        auc: The area under the ROC curve
    """
    # Combine the true labels and predicted scores
    data = list(zip(y_scores, y_true))
    
    # Sort by predicted scores in descending order
    data.sort(key=lambda x: -x[0])
    
    # Calculate the number of positive and negative examples
    num_pos = sum(y_true)
    num_neg = len(y_true) - num_pos
    
    if num_pos == 0 or num_neg == 0:
        raise ValueError("Need both positive and negative examples to compute AUC")
    
    # Initialize variables
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0
    tp = 0
    fp = 0
    
    # Iterate through the sorted data
    for i in range(len(data)):
        _, label = data[i]
        
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        # Calculate current TPR and FPR
        tpr = tp / num_pos
        fpr = fp / num_neg
        
        # Add area of the trapezoid
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        
        prev_fpr = fpr
        prev_tpr = tpr
    
    return auc

# Example usage
if __name__ == "__main__":
    # True binary labels
    y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    
    # Predicted probabilities for class 1
    y_scores = [0.1, 0.3, 0.3, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    auc = calculate_auc(y_true, y_scores)
    print(f"AUC: {auc:.4f}")

    from sklearn.metrics import roc_auc_score
    print(f"Sklearn AUC: {roc_auc_score(y_true, y_scores):.4f}")