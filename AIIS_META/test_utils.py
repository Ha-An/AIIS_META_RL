import numpy as np
from Utils import utils


def test_discount_cumsum():
    x = np.array([1, 1, 1], dtype=np.float32)
    out = utils.discount_cumsum(x, gamma=1.0)
    assert np.allclose(out, [3, 2, 1])


def test_compute_gae():
    rewards = np.array([1, 1, 1], dtype=np.float32)
    values = np.array([0, 0, 0, 0], dtype=np.float32)  # T+1
    adv = utils.compute_gae(rewards, values, gamma=1.0, lam=1.0)
    assert adv.shape == (3,)
    assert adv[0] >= adv[1] >= adv[2]


def test_normalize():
    x = np.array([1.0, 2.0, 3.0])
    norm = utils.normalize(x)
    assert np.isclose(np.mean(norm), 0, atol=1e-6)
    assert np.isclose(np.std(norm), 1, atol=1e-6)
