from unittest import TestCase
# from sklearn.datasets import make_regression
from gbt0 import GradientBoostedTree
import numpy as np

class TestFeature(TestCase):
    def test_feature(self):
        rng = np.random.default_rng(seed = 42)
        X = rng.random(size=(3, 100))
        y = rng.random(size=3)
        gbt = GradientBoostedTree()
        gbt.fit(X,y)
        pred = gbt.predict(X)
        np.testing.assert_allclose(y, pred, rtol=1e-5)