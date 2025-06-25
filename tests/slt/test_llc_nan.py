import numpy as np
import pytest
import torch
import torch.nn as nn
from devinterp.optim.sgld import SGLD
from devinterp.slt.llc import LLCEstimator, OnlineLLCEstimator
from devinterp.slt.sampler import estimate_learning_coeff, sample
from devinterp.utils import default_nbeta, evaluate_mse, get_init_loss_multi_batch


def test_llc_estimator_nan_error():
    llc_estimator = LLCEstimator(num_chains=3, num_draws=100, nbeta=10, init_loss=1.0)

    with pytest.raises(RuntimeError, match="NaN detected in loss at chain 0, draw 0"):
        llc_estimator.update(0, 0, torch.tensor(np.nan))

    llc_estimator = OnlineLLCEstimator(
        num_chains=3, num_draws=100, init_loss=1.0, nbeta=10
    )

    with pytest.raises(RuntimeError, match="NaN detected in loss at chain 0, draw 0"):
        llc_estimator.update(0, 0, torch.tensor(np.nan))


def test_sampling_nan_error(DummyNaNModel):
    model = DummyNaNModel()

    # Create a simple dataset and dataloader
    inputs = torch.randn(100, 2)  # 100 samples of 2D data
    dataset = torch.utils.data.TensorDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)

    # Define loss function
    def loss_fn(model, data):
        x = data[0]  # Unpack the single input tensor
        return model(x).mean()

    with pytest.raises(RuntimeError, match="NaN detected in loss at chain 0, draw 97"):
        estimate_learning_coeff(
            model=model,
            loader=loader,
            evaluate=loss_fn,
            num_draws=100,
            num_chains=3,
            sampling_method_kwargs={"nbeta": 10},
            device="cpu",
        )


def test_llc_nan_model(generated_linedot_normalcrossing_dataset, Polynomial):
    seed = 42
    torch.manual_seed(seed)
    model = Polynomial([2, 2])

    train_dataloader, _, _, _ = generated_linedot_normalcrossing_dataset
    num_chains = 1
    num_draws = 1_000
    sample_point = [[0.0 for _ in range(2)]]
    model.weights = nn.Parameter(
        torch.tensor(sample_point, dtype=torch.float32, requires_grad=True)
    )
    init_loss = get_init_loss_multi_batch(
        train_dataloader, num_chains, model, evaluate_mse, device="cpu"
    )
    llc_estimator = LLCEstimator(
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss,
    )
    with pytest.raises(RuntimeError, match="NaN detected in loss at chain 0, draw 4"):
        sample(
            model,
            train_dataloader,
            evaluate=evaluate_mse,
            sampling_method_kwargs=dict(lr=1000, nbeta=1000.0),
            sampling_method=SGLD,
            num_chains=num_chains,
            num_draws=num_draws,
            callbacks=[llc_estimator],
            verbose=False,
        )
