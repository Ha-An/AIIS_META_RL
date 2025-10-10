import torch
from Agents.base import BasePolicy


class DummyPolicy(BasePolicy):
    def act(self, obs, deterministic=False):
        return torch.zeros(self.out_dim), {"logp": torch.tensor(0.0)}

    def log_prob(self, obs, actions, params=None):
        return torch.tensor(0.0)


def test_base_policy_act_logprob():
    policy = DummyPolicy(obs_dim=4, out_dim=2)
    obs = torch.randn(4)
    action, info = policy.act(obs)
    assert action.shape == (2,)
    assert "logp" in info

    logp = policy.log_prob(obs, action)
    assert torch.is_tensor(logp)
