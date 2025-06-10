import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import platform
from devinterp.optim import SGLD, SGMCMC
from devinterp.slt.llc import LLCEstimator
from devinterp.slt.sampler import sample
from devinterp.utils import default_nbeta, evaluate_mse, get_init_loss_multi_batch
from torch.utils.data import DataLoader, TensorDataset

# Test configuration constants
SNAPSHOT_DRAWS = 5
FULL_SAMPLING_DRAWS = 1000
NUM_CHAINS = 5
RANDOM_SEED = 42


def _do_ordinality_sampling(
    model, train_dataloader, sampling_method, lr, sample_points, num_draws
):
    """Perform MCMC sampling to estimate LLC for ordinality test.

    Args:
        model: Model to sample from
        train_dataloader: DataLoader for training data
        sampling_method: SGLD or SGMCMC sampling method
        lr: Learning rate
        sample_points: List of sample points to test
        num_draws: Number of MCMC draws to perform

    Returns:
        List of LLC means for each sample point
    """
    torch.manual_seed(RANDOM_SEED)

    llcs = []
    for sample_point in sample_points:
        model.weights = nn.Parameter(
            torch.tensor(sample_point, dtype=torch.float32, requires_grad=True)
        )
        init_loss = get_init_loss_multi_batch(
            train_dataloader, NUM_CHAINS, model, evaluate_mse, device="cpu"
        )
        llc_estimator = LLCEstimator(
            num_chains=NUM_CHAINS,
            num_draws=num_draws,
            nbeta=default_nbeta(train_dataloader),
            init_loss=init_loss,
        )

        sample(
            model,
            train_dataloader,
            evaluate=evaluate_mse,
            sampling_method_kwargs=dict(
                lr=lr,
                bounding_box_size=0.5,
                nbeta=default_nbeta(train_dataloader),
                # to prevent accidental movement from [1, 0, ...] to origin
            ),
            sampling_method=sampling_method,
            num_chains=NUM_CHAINS,
            num_draws=num_draws,
            callbacks=[llc_estimator],
            verbose=False,
            seed=RANDOM_SEED,
        )
        llcs.append(llc_estimator.get_results()["llc/mean"])

    return llcs


def _test_ordinality_accuracy(
    model, train_dataloader, sampling_method, lr, sample_points, model_name, dim
):
    """Test LLC ordinality with full sampling."""
    llcs = _do_ordinality_sampling(
        model,
        train_dataloader,
        sampling_method,
        lr,
        sample_points,
        num_draws=FULL_SAMPLING_DRAWS,
    )

    assert (
        np.diff(llcs) >= 0
    ).all(), f"Ordinality not preserved for sampler {sampling_method} on {dim}-d {model_name}: llcs {llcs} are not in ascending order."


@pytest.mark.skipif(
    platform.machine() != "x86_64",
    reason=f"Differences in results between ARM and x86_64. Your arch is {platform.machine()}",
)
@pytest.mark.parametrize("sampling_method", [SGLD, SGMCMC.sgld])
@pytest.mark.parametrize("model_name", ["Polynomial", "LinePlusDot"])
@pytest.mark.parametrize("dim", [2, 10])
def test_linedot_normal_crossing(
    generated_linedot_normalcrossing_dataset,
    sampling_method,
    model_name,
    dim,
    request,
    snapshot,
    is_snapshot_update,
):
    torch.manual_seed(RANDOM_SEED)
    Model = request.getfixturevalue(model_name)
    if model_name == "Polynomial":
        model = Model([2 for _ in range(dim)])
    else:
        model = Model(dim)
    train_dataloader, _, _, _ = generated_linedot_normalcrossing_dataset
    lr = (
        0.0001 / dim
    )  # to account for smaller steps in higher D. might not work well for SGNHT?

    sample_points = [
        [0.0 for _ in range(dim)],
        [0.0 if i == dim - 1 else 1.0 for i in range(dim)],
    ]

    # Run snapshot test first
    llcs = _do_ordinality_sampling(
        model,
        train_dataloader,
        sampling_method,
        lr,
        sample_points,
        num_draws=SNAPSHOT_DRAWS,
    )

    # Run verification test when updating snapshots
    if is_snapshot_update:
        _test_ordinality_accuracy(
            model, train_dataloader, sampling_method, lr, sample_points, model_name, dim
        )

    # Test against snapshot
    assert llcs == snapshot
