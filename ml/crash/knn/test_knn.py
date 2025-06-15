from unittest import TestCase
from knn1 import KNN

class TestMyCase(TestCase):
    def test_my_feature(self):
        # Sample dataset
        X_train = [
            [2, 4], [4, 6], [4, 4], [4, 2],
            [6, 4], [6, 2], [3, 5], [5, 5]
        ]
        y_train = ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'B']

        # Create and train the KNN classifier
        knn = KNN(k=3)
        knn.fit(X_train, y_train)

        # Test samples
        X_test = [
            [5, 3],
            [3, 4]
        ]

        # Make predictions
        predictions = knn.predict(X_test)
        print("Predictions:", predictions)  # Output: ['B', 'A']

        self.assertEqual(predictions, ['B', 'A'])