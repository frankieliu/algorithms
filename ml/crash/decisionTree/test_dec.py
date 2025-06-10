from unittest import TestCase
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from dec1 import DecisionTreeClassifier

class TestMyCase(TestCase):
    def test_my_feature(self):

        # Load data
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,test_size=0.2, random_state=42)

        # Train
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X_train, y_train)

        # Predict
        pred = clf.predict(X_test)
        print("Predictions:", pred)
        print("Actual:", y_test)
        print("Diff:", abs(y_test - pred))
        self.assertLessEqual(sum(abs(y_test-pred)), 1)
