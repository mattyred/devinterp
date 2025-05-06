import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# --- Toy model fixtures ---
@pytest.fixture
def Polynomial():
    class _Poly(nn.Module):
        def __init__(self, powers=(1, 1)):
            super().__init__()
            self.powers = torch.tensor(powers, dtype=torch.float32)
            self.weights = nn.Parameter(torch.zeros_like(self.powers))

        def forward(self, x):
            return x * torch.prod(self.weights**self.powers)

    return _Poly


@pytest.fixture
def LinePlusDot():
    class _LPD(nn.Module):
        def __init__(self, dim=2):
            super().__init__()
            self.weights = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

        def forward(self, x):
            return x * (self.weights[0] - 1) * (self.weights.pow(2).sum().pow(2))

    return _LPD


@pytest.fixture
def ReducedRankRegressor():
    class _RRR(nn.Module):
        def __init__(self, m, h, n):
            super().__init__()
            self.fc1 = nn.Linear(m, h, bias=False)
            self.fc2 = nn.Linear(h, n, bias=False)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    return _RRR


@pytest.fixture
def DummyNaNModel():
    class _DN(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
            self.counter = 0
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.fill_(0.0)

        def forward(self, x):
            self.counter += 1
            if self.counter > 100:
                with torch.no_grad():
                    self.linear.weight.fill_(float("inf"))
            return self.linear(x)

    return _DN


# --- Shared dataset fixtures ---
@pytest.fixture
def generated_normalcrossing_dataset():
    """Shared dataset: normal input with small Gaussian noise."""
    torch.manual_seed(42)
    np.random.seed(42)
    sigma = 0.25
    num_samples = 1000
    x = torch.normal(0, 2, size=(num_samples,))
    y = sigma * torch.normal(0, 1, size=(num_samples,))
    train_data = TensorDataset(x, y)
    train_dataloader = DataLoader(train_data, batch_size=num_samples, shuffle=True)
    return train_dataloader, train_data, x, y


@pytest.fixture
def generated_linedot_normalcrossing_dataset():
    """Shared dataset: single-dim x, noise y for line-plus-dot tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    sigma = 0.25
    num_samples = 1000
    x = torch.normal(0, 2, size=(num_samples,))
    y = sigma * torch.normal(0, 1, size=(num_samples,))
    train_data = TensorDataset(x, y)
    train_dataloader = DataLoader(train_data, batch_size=num_samples, shuffle=True)
    return train_dataloader, train_data, x, y


# --- LLCEstimator runner fixture ---
from devinterp.utils import default_nbeta, evaluate_mse, get_init_loss_multi_batch
from devinterp.slt.llc import LLCEstimator
from devinterp.slt.sampler import sample as _sample


@pytest.fixture
def run_llc_estimator():
    """
    Returns a callable to run LLCEstimator + sampling and return the estimator.
    Usage:
        estimator = run_llc_estimator(
            model, loader, sampling_method, lr,
            nbeta=None, num_chains=1, num_draws=100, seed=42,
            sampling_method_kwargs=None,
            **llc_init_kwargs
        )
    """

    def _run(
        model,
        loader,
        sampling_method,
        lr,
        nbeta=None,
        num_chains=1,
        num_draws=100,
        seed=42,
        sampling_method_kwargs=None,
        **llc_init_kwargs,
    ):
        if nbeta is None:
            nbeta = default_nbeta(loader)
        init_loss = get_init_loss_multi_batch(
            loader, num_chains, model, evaluate_mse, device="cpu"
        )
        estimator = LLCEstimator(
            num_chains=num_chains,
            num_draws=num_draws,
            nbeta=nbeta,
            init_loss=init_loss,
            **llc_init_kwargs,
        )
        torch.manual_seed(seed)
        sm_kwargs = {"lr": lr, "nbeta": nbeta}
        if sampling_method_kwargs:
            sm_kwargs.update(sampling_method_kwargs)
        _sample(
            model,
            loader,
            evaluate=evaluate_mse,
            sampling_method_kwargs=sm_kwargs,
            sampling_method=sampling_method,
            num_chains=num_chains,
            num_draws=num_draws,
            callbacks=[estimator],
            verbose=False,
            seed=seed,
        )
        return estimator

    return _run
