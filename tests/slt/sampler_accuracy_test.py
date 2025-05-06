import numpy as np
import pytest
import torch
import torch.nn.functional as F
from devinterp.optim import SGLD, SGMCMC
from devinterp.slt.llc import LLCEstimator, OnlineLLCEstimator
from devinterp.slt.sampler import sample
from devinterp.utils import default_nbeta, evaluate_mse, get_init_loss_multi_batch
from torch.utils.data import DataLoader, TensorDataset


TRUE_LCS_PER_POWER = [
    [[0, 1], 0.5],
    [[1, 1], 0.5],
    [[0, 2], 0.25],
    [[1, 2], 0.25],
    [[2, 2], 0.25],
    [[0, 3], 0.166],
    [[1, 3], 0.166],
    [[2, 3], 0.166],
    [[3, 3], 0.166],
]


@pytest.mark.slow
@pytest.mark.parametrize("sampling_method", [SGLD, SGMCMC.sgld])
@pytest.mark.parametrize("powers, true_lc", TRUE_LCS_PER_POWER)
def test_accuracy_normalcrossing(
    generated_normalcrossing_dataset,
    sampling_method,
    powers,
    true_lc,
    Polynomial,
    run_llc_estimator,
):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Polynomial(powers)
    train_dataloader, train_data, _, _ = generated_normalcrossing_dataset
    lr = 0.0004
    num_chains = 10
    num_draws = 5_000
    nbeta = default_nbeta(len(train_data))
    estimator = run_llc_estimator(
        model,
        train_dataloader,
        sampling_method,
        lr=lr,
        nbeta=nbeta,
        num_chains=num_chains,
        num_draws=num_draws,
        seed=seed,
    )
    llc_mean = estimator.get_results()["llc/mean"]
    llc_std_dev = estimator.get_results()["llc/std"]
    assert (
        llc_mean - 3.5 * llc_std_dev < true_lc < llc_mean + 3.5 * llc_std_dev
    ), f"LLC mean {llc_mean:.3f} +- {3.5*llc_std_dev:.3f} does not contain true value {true_lc:.3f} for powers {powers} using {sampling_method}"
