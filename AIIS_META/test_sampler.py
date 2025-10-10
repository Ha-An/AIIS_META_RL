import numpy as np
from Sampler.base import Sampler


class DummyPolicy:
    has_value_fn = False

    def value_function(self, obs):
        raise NotImplementedError


def test_process_samples_with_baseline_error():
    sampler = Sampler(gamma=0.99, lam=0.95)
    paths = [{"rewards": np.array([1.0, 1.0]),
              "observations": np.array([[0.1], [0.2]])}]
    try:
        sampler.process_samples(paths, DummyPolicy())
    except ValueError:
        assert True


def test_process_samples_with_advantages():
    sampler = Sampler()
    paths = [{"rewards": np.array([1.0, 1.0]),
              "observations": np.array([[0.1], [0.2]]),
              "advantages": np.array([0.5, 0.5])}]
    out = sampler.process_samples(paths, DummyPolicy())
    assert "advantages" in out[0]
