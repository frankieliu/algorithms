from unittest import TestCase
from auc_simple2 import calculate_auc

# Example usage
class MyTestCase(TestCase):
    def test_auc(self):

        # True binary labels
        y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
        
        # Predicted probabilities for class 1
        y_scores = [0.1, 0.3, 0.3, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        auc = calculate_auc(y_true, y_scores)
        print(f"AUC: {auc:.4f}")

        from sklearn.metrics import roc_auc_score
        sk_auc = roc_auc_score(y_true, y_scores)
        print(f"Sklearn AUC: {sk_auc:.4f}")

        self.assertAlmostEqual(auc, sk_auc, places=7)