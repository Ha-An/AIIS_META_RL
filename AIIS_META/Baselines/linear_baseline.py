from .base import Baseline
import numpy as np
import torch

class LinearBaseline(Baseline):
    """
    Abstract class providing the functionality for fitting a linear baseline
    Don't instantiate this class. Instead use LinearFeatureBaseline or LinearTimeBaseline
    """

    def __init__(self, reg_coeff=1e-5):
        super(LinearBaseline, self).__init__()
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def predict(self, path, device):
        """
        Abstract Class for the LinearFeatureBaseline and the LinearTimeBaseline
        Predicts the linear reward baselines estimates for a provided trajectory / path.
        If the baseline is not fitted - returns zero baseline

        Args:
           path (dict): dict of lists/numpy array containing trajectory / path information
                 such as "observations", "rewards", ...

        Returns:
             (np.ndarray): numpy array of the same length as paths["observations"] specifying the reward baseline

        """
        if self._coeffs is None:
            return torch.zeros(len(path["observations"]))
        
        return torch.matmul(self._features(path, device), (self._coeffs.to(device)))

    def get_param_values(self, **tags):
        """
        Returns the parameter values of the baseline object

        Returns:
            numpy array of linear_regression coefficients

        """
        return self._coeffs

    def set_params(self, value, **tags):
        """
        Sets the parameter values of the baseline object

        Args:
            value: numpy array of linear_regression coefficients

        """
        self._coeffs = value

    def fit(self, paths, target_key='returns'):
        """
        Fits the linear baseline model with the provided paths via damped least squares

        Args:
            paths (list): list of paths
            target_key (str): path dictionary key of the target that shall be fitted (e.g. "returns")

        """
        assert all([target_key in path.keys() for path in paths])

        
        target = torch.cat([torch.asarray(path[target_key]) for path in paths], axis = 0)
        featmat = torch.cat([self._features(path, target.device) for path in paths], axis=0)
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = torch.linalg.lstsq(
                torch.matmul(featmat.T, featmat) + reg_coeff * torch.eye(featmat.shape[1], device=featmat.device, dtype=featmat.dtype),
                featmat.T @ target,
                rcond=-1
            )[0]
            if not torch.isnan(self._coeffs).any().item():
                break
            reg_coeff *= 10
    def _features(self, path):
        raise NotImplementedError("this is an abstract class, use either LinearFeatureBaseline or LinearTimeBaseline")


class LinearFeatureBaseline(LinearBaseline):
    """
    Linear (polynomial) time-state dependent return baseline model
    (see. Duan et al. 2016, "Benchmarking Deep Reinforcement Learning for Continuous Control", ICML)

    Fits the following linear model

    reward = b0 + b1*obs + b2*obs^2 + b3*t + b4*t^2+  b5*t^3

    Args:
        reg_coeff: list of paths

    """
    def __init__(self, reg_coeff=1e-5,):
        super(LinearFeatureBaseline, self).__init__()
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def _features(self, path, device):
        
        obs = torch.asarray(path["observations"], device=device)
        path_length = len(path["observations"])
        time_step = torch.arange(path_length, dtype=torch.float32, device=device).reshape(-1, 1) / 100.0
        return torch.cat([
                obs,
                obs ** 2,
                time_step,
                time_step ** 2,
                time_step ** 3,
                torch.ones((path_length, 1), dtype=obs.dtype, device=device)
            ], dim=1)


class LinearTimeBaseline(LinearBaseline):
    """
    Linear (polynomial) time-dependent reward baseline model

    Fits the following linear model

    reward = b0 + b3*t + b4*t^2+  b5*t^3

    Args:
        reg_coeff: list of paths

    """

    def _features(self, path):
        path_length = len(path["observations"])
        time_step = np.arange(path_length).reshape(-1, 1) / 100.0
        return np.concatenate([time_step, time_step ** 2, time_step ** 3, np.ones((path_length, 1))],
                              axis=1)