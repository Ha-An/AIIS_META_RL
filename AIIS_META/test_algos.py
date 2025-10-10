import torch
from Algos.MAML.base import *

class DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class DummyMAML(MAML_BASE):
    def inner_obj(self, batch, params):
        return torch.tensor(0.0, requires_grad=True)

    def outer_obj(self, batch, params):
        return torch.tensor(0.0, requires_grad=True)

    def step_kl(self, batch, params):
        return torch.tensor(0.0)


def test_maml_learn():
    policy = DummyPolicy()
    maml = DummyMAML(env=None, policy=policy, num_tasks=2)
    # sampler mock
    class DummySampler:
        def obtain_samples(self, params_list):
            return [[{"observations": [], "actions": [], "rewards": []}]] * 2
    sampler = DummySampler()
    maml.learn(sampler, total_iters=1)
