import numpy as np
from Baselines.linear_baseline import LinearFeatureBaseline
from Baselines.zero_baseline import ZeroBaseline


def test_linear_baseline_fit_predict():
    baseline = LinearFeatureBaseline()
    paths = [{"observations": np.array([[1, 2], [3, 4]]),
              "returns": np.array([1.0, 2.0])}]
    baseline.fit(paths)
    preds = baseline.predict(paths[0])
    assert preds.shape == (2,)


def test_zero_baseline():
    baseline = ZeroBaseline()
    path = {"observations": np.array([[1, 2], [3, 4]])}
    preds = baseline.predict(path)
    assert np.all(preds == 0)
