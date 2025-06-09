import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_scores):
    # Sort by descending score
    data = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    
    # Initialize
    tpr, fpr, thresholds = [], [], []
    tp = fp = 0
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    
    # Iterate through thresholds
    prev_score = None
    for score, label in data:
        if score != prev_score:
            thresholds.append(score)
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)
            prev_score = score
        if label == 1:
            tp += 1
        else:
            fp += 1
    
    # Add (0,0) to complete the curve
    tpr.append(0)
    fpr.append(0)
    
    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.5)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example
y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
y_scores = [0.1, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
plot_roc_curve(y_true, y_scores)