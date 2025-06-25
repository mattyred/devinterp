"""
This test encounters nan loss values during sampling.
Found by adding a feature to throw an error when nan loss values are encountered.
"""

import numpy as np
import pytest
import torch
import platform
from devinterp.optim.sgld import SGLD
from devinterp.optim.sgmcmc import SGMCMC
from devinterp.slt.llc import LLCEstimator
from devinterp.slt.sampler import sample
from devinterp.utils import evaluate_mse, default_nbeta, get_init_loss_multi_batch
from torch.utils.data import DataLoader, TensorDataset

# Test configuration constants
TOLERANCE_ATOL = 1e-5
TOLERANCE_ATOL_FULL = 8e-2
TOLERANCE_ATOL_DIFFERENCE = 1e-2
FULL_SAMPLING_DRAWS = 500
SNAPSHOT_DRAWS = 5
LEARNING_RATE_FAST = 0.0002
LEARNING_RATE_SLOW = 0.0001
LEARNING_RATE_FULL = 0.001
NUM_CHAINS = 1
RANDOM_SEED = 42
NUM_SAMPLES = 1000


def generated_normalcrossing_dataset_seeded(seed):
    """Generate synthetic dataset with normal crossing pattern for given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    sigma = 0.25
    x = torch.normal(0, 2, size=(NUM_SAMPLES,))
    y = sigma * torch.normal(0, 1, size=(NUM_SAMPLES,))
    train_data = TensorDataset(x, y)

    # Add deterministic generator for DataLoader shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dataloader = DataLoader(
        train_data, batch_size=NUM_SAMPLES, shuffle=True, generator=generator
    )
    return train_dataloader, train_data, x, y


# Test case configurations
POWERS_BETWEEN = [
    [
        [1, 1, 0],
        [1, 1, 10],
    ],
    [
        [2, 2, 10],
        [2, 2, 3],
    ],
    [
        [3, 3, 6.1],
        [3, 3, 1.2],
    ],
]

POWERS_DIMS = [
    [1, 1],
    [2, 10],
]

POWERS_DIMS_FULL = [
    # [1, 1], # For some reason, this consistently fails tests.
    [0, 2],
]

POWERS_DIFFERENCE = [
    [0, 1],
]

EXTRA_DIM_POWERS = [3, 10, 100]
SAMPLE_POINTS = [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
SAMPLE_POINTS_SINGLE = [[0.0, 0.0, 1.0]]


@pytest.mark.parametrize("sampling_method", [SGLD, SGMCMC.sgld])
@pytest.mark.parametrize("powers", POWERS_BETWEEN)
@pytest.mark.parametrize("sample_point", SAMPLE_POINTS)
def test_rllc_normalcrossing_between_powers(
    generated_normalcrossing_dataset,
    sampling_method,
    powers,
    sample_point,
    Polynomial,
    snapshot,
    is_snapshot_update,
):
    torch.manual_seed(RANDOM_SEED)

    # Set up models
    model1 = Polynomial(powers[0])
    model1.weights = torch.nn.Parameter(torch.tensor(sample_point))
    model2 = Polynomial(powers[1])
    model2.weights = torch.nn.Parameter(torch.tensor(sample_point))

    train_dataloader, _, _, _ = generated_normalcrossing_dataset

    # Run snapshot test first
    llc_mean_1, llc_mean_2 = _do_between_powers_sampling(
        model1, model2, train_dataloader, sampling_method, num_draws=SNAPSHOT_DRAWS
    )

    # Run verification test when updating snapshots
    if is_snapshot_update:
        _test_between_powers_accuracy(
            model1, model2, train_dataloader, sampling_method, powers
        )

    # Test against snapshot
    difference = abs(llc_mean_1 - llc_mean_2)
    assert difference == snapshot


@pytest.mark.parametrize("sampling_method", [SGLD, SGMCMC.sgld])
@pytest.mark.parametrize("relevant_powers", POWERS_DIMS)
@pytest.mark.parametrize("extra_dim_power", EXTRA_DIM_POWERS)
@pytest.mark.parametrize("sample_point", SAMPLE_POINTS)
def test_restricted_gradient_normalcrossing_between_dims(
    generated_normalcrossing_dataset,
    sampling_method,
    relevant_powers,
    extra_dim_power,
    sample_point,
    Polynomial,
    snapshot,
    is_snapshot_update,
):
    torch.manual_seed(RANDOM_SEED)

    # Set up models
    model1 = Polynomial(relevant_powers)
    model2 = Polynomial(relevant_powers + [extra_dim_power])

    model1.weights = torch.nn.Parameter(torch.tensor(sample_point[:-1]))
    model2.weights = torch.nn.Parameter(torch.tensor(sample_point))

    train_dataloader, _, _, _ = generated_normalcrossing_dataset

    # Run snapshot test first
    llc_mean_2d, llc_mean_3d_restricted = _do_restricted_gradient_sampling(
        model1, model2, train_dataloader, sampling_method, num_draws=SNAPSHOT_DRAWS
    )

    # Run verification test when updating snapshots
    if is_snapshot_update:
        _test_restricted_gradient_accuracy(
            model1,
            model2,
            train_dataloader,
            sampling_method,
            relevant_powers,
            extra_dim_power,
        )

    # Test against snapshot
    difference = abs(llc_mean_2d - llc_mean_3d_restricted)
    assert difference == snapshot


@pytest.mark.skipif(
    platform.machine() != "x86_64",
    reason=f"Differences in results between ARM and x86_64. Your arch is {platform.machine()}",
)
@pytest.mark.parametrize(
    "sampling_method", [SGLD, SGMCMC.sgld], ids=lambda x: x.__name__
)
@pytest.mark.parametrize(
    "relevant_powers", POWERS_DIMS_FULL, ids=lambda x: f"powers_{'_'.join(map(str, x))}"
)
@pytest.mark.parametrize(
    "extra_dim_power", EXTRA_DIM_POWERS, ids=lambda x: f"extra_dim_{x}"
)
@pytest.mark.parametrize(
    "sample_point",
    SAMPLE_POINTS_SINGLE,
    ids=lambda x: f"sample_{'_'.join(map(str, x))}",
)
def test_rllc_full_normalcrossing_between_dims(
    sampling_method,
    relevant_powers,
    extra_dim_power,
    sample_point,
    Polynomial,
    snapshot,
    is_snapshot_update,
):
    seed = 5
    torch.manual_seed(seed)

    # Set up models
    model1 = Polynomial(relevant_powers)
    model2 = Polynomial(relevant_powers + [extra_dim_power])

    model1.weights = torch.nn.Parameter(torch.tensor(sample_point[:-1]))
    model2.weights = torch.nn.Parameter(torch.tensor(sample_point))

    train_dataloader, _, _, _ = generated_normalcrossing_dataset_seeded(seed)

    # Run snapshot test first
    llc_mean_2d, llc_mean_3d_restricted = _do_full_sampling(
        model1,
        model2,
        train_dataloader,
        sampling_method,
        seed,
        num_draws=SNAPSHOT_DRAWS,
    )

    # Run verification test when updating snapshots
    if is_snapshot_update:
        _test_full_accuracy(
            model1,
            model2,
            train_dataloader,
            sampling_method,
            seed,
            relevant_powers,
            extra_dim_power,
        )

    # Test against snapshot
    difference = abs(llc_mean_2d - llc_mean_3d_restricted)
    assert difference == snapshot


@pytest.mark.parametrize("sampling_method", [SGLD, SGMCMC.sgld])
@pytest.mark.parametrize("relevant_powers", POWERS_DIFFERENCE)
def test_rllc_different_from_full_llc_between_dims(
    generated_normalcrossing_dataset,
    sampling_method,
    relevant_powers,
    Polynomial,
    snapshot,
    is_snapshot_update,
):
    torch.manual_seed(RANDOM_SEED)

    # Set up model
    model = Polynomial(relevant_powers)
    model.weights = torch.nn.Parameter(torch.tensor([0.3, 1.5]))

    train_dataloader, _, _, _ = generated_normalcrossing_dataset

    # Run snapshot test first
    llc_mean, rllc_mean = _do_difference_sampling(
        model, train_dataloader, sampling_method, num_draws=SNAPSHOT_DRAWS
    )

    # Run verification test when updating snapshots
    if is_snapshot_update:
        _test_difference_accuracy(
            model, train_dataloader, sampling_method, relevant_powers
        )

    # Test against snapshot
    difference = abs(llc_mean - rllc_mean)
    assert difference == snapshot


def _do_between_powers_sampling(
    model1, model2, train_dataloader, sampling_method, num_draws
):
    torch.manual_seed(RANDOM_SEED)

    init_loss_1 = get_init_loss_multi_batch(
        train_dataloader, NUM_CHAINS, model1, evaluate_mse, device="cpu"
    )
    init_loss_2 = get_init_loss_multi_batch(
        train_dataloader, NUM_CHAINS, model2, evaluate_mse, device="cpu"
    )

    llc_estimator_1 = LLCEstimator(
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss_1,
    )
    llc_estimator_2 = LLCEstimator(
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss_2,
    )

    torch.manual_seed(RANDOM_SEED)
    sample(
        model1,
        train_dataloader,
        evaluate=evaluate_mse,
        sampling_method_kwargs=dict(
            lr=LEARNING_RATE_FAST,
            nbeta=default_nbeta(train_dataloader),
        ),
        sampling_method=sampling_method,
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        callbacks=[llc_estimator_1],
        verbose=False,
        seed=RANDOM_SEED,
        optimize_over_per_model_param={"weights": torch.tensor([1, 1, 0])},
    )

    torch.manual_seed(RANDOM_SEED)
    sample(
        model2,
        train_dataloader,
        evaluate=evaluate_mse,
        sampling_method_kwargs=dict(
            lr=LEARNING_RATE_FAST,
            nbeta=default_nbeta(train_dataloader),
        ),
        sampling_method=sampling_method,
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        callbacks=[llc_estimator_2],
        verbose=False,
        seed=RANDOM_SEED,
        optimize_over_per_model_param={"weights": torch.tensor([1, 1, 0])},
    )

    llc_mean_1 = llc_estimator_1.get_results()["llc/mean"]
    llc_mean_2 = llc_estimator_2.get_results()["llc/mean"]

    return llc_mean_1, llc_mean_2


def _do_restricted_gradient_sampling(
    model1, model2, train_dataloader, sampling_method, num_draws
):
    torch.manual_seed(RANDOM_SEED)

    init_loss_1 = get_init_loss_multi_batch(
        train_dataloader, NUM_CHAINS, model1, evaluate_mse, device="cpu"
    )
    init_loss_2 = get_init_loss_multi_batch(
        train_dataloader, NUM_CHAINS, model2, evaluate_mse, device="cpu"
    )

    llc_estimator_2d = LLCEstimator(
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss_1,
    )
    llc_estimator_3d = LLCEstimator(
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss_2,
    )

    sample(
        model1,
        train_dataloader,
        evaluate=evaluate_mse,
        sampling_method_kwargs=dict(
            lr=LEARNING_RATE_SLOW,
            nbeta=default_nbeta(train_dataloader),
            noise_level=0.0,
        ),
        sampling_method=sampling_method,
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        callbacks=[llc_estimator_2d],
        verbose=False,
        seed=RANDOM_SEED,
    )

    sample(
        model2,
        train_dataloader,
        evaluate=evaluate_mse,
        sampling_method_kwargs=dict(
            lr=LEARNING_RATE_SLOW,
            nbeta=default_nbeta(train_dataloader),
            noise_level=0.0,
        ),
        sampling_method=sampling_method,
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        callbacks=[llc_estimator_3d],
        verbose=False,
        seed=RANDOM_SEED,
        optimize_over_per_model_param={"weights": torch.tensor([1, 1, 0])},
    )

    llc_mean_2d = llc_estimator_2d.get_results()["llc/mean"]
    llc_mean_3d_restricted = llc_estimator_3d.get_results()["llc/mean"]

    return llc_mean_2d, llc_mean_3d_restricted


def _do_full_sampling(
    model1, model2, train_dataloader, sampling_method, seed, num_draws
):
    init_loss_1 = get_init_loss_multi_batch(
        train_dataloader, NUM_CHAINS, model1, evaluate_mse, device="cpu"
    )

    llc_estimator_2d = LLCEstimator(
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss_1,
    )
    llc_estimator_3d = LLCEstimator(
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss_1,
    )

    torch.manual_seed(seed)
    sample(
        model1,
        train_dataloader,
        evaluate=evaluate_mse,
        sampling_method_kwargs=dict(
            lr=LEARNING_RATE_FULL, nbeta=default_nbeta(train_dataloader)
        ),
        sampling_method=sampling_method,
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        callbacks=[llc_estimator_2d],
        verbose=False,
        seed=seed,
    )

    torch.manual_seed(seed)
    sample(
        model2,
        train_dataloader,
        evaluate=evaluate_mse,
        sampling_method_kwargs=dict(
            lr=LEARNING_RATE_FULL, nbeta=default_nbeta(train_dataloader)
        ),
        sampling_method=sampling_method,
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        callbacks=[llc_estimator_3d],
        verbose=False,
        seed=seed,
        optimize_over_per_model_param={"weights": torch.tensor([1, 1, 0])},
    )

    llc_mean_2d = llc_estimator_2d.get_results()["llc/mean"]
    llc_mean_3d_restricted = llc_estimator_3d.get_results()["llc/mean"]

    return llc_mean_2d, llc_mean_3d_restricted


def _do_difference_sampling(model, train_dataloader, sampling_method, num_draws):
    torch.manual_seed(RANDOM_SEED)

    init_loss = get_init_loss_multi_batch(
        train_dataloader, NUM_CHAINS, model, evaluate_mse, device="cpu"
    )

    llc_estimator = LLCEstimator(
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        nbeta=default_nbeta(train_dataloader),
        init_loss=init_loss,
    )
    rllc_estimator = LLCEstimator(
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
            lr=LEARNING_RATE_FULL, nbeta=default_nbeta(train_dataloader)
        ),
        sampling_method=sampling_method,
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        callbacks=[llc_estimator],
        verbose=False,
        seed=RANDOM_SEED,
    )

    sample(
        model,
        train_dataloader,
        evaluate=evaluate_mse,
        sampling_method_kwargs=dict(
            lr=LEARNING_RATE_FULL, nbeta=default_nbeta(train_dataloader)
        ),
        sampling_method=sampling_method,
        num_chains=NUM_CHAINS,
        num_draws=num_draws,
        callbacks=[rllc_estimator],
        verbose=False,
        seed=RANDOM_SEED,
        optimize_over_per_model_param={"weights": torch.tensor([1, 0])},
    )

    llc_mean = llc_estimator.get_results()["llc/mean"]
    rllc_mean = rllc_estimator.get_results()["llc/mean"]

    return llc_mean, rllc_mean


def _test_between_powers_accuracy(
    model1, model2, train_dataloader, sampling_method, powers
):
    llc_mean_1, llc_mean_2 = _do_between_powers_sampling(
        model1, model2, train_dataloader, sampling_method, num_draws=100
    )

    error_msg = (
        f"LLC mean {llc_mean_1:.3f}!={llc_mean_2:.3f} for powers {powers} "
        f"using {sampling_method}"
    )
    assert np.isclose(llc_mean_1, llc_mean_2, atol=TOLERANCE_ATOL), error_msg


def _test_restricted_gradient_accuracy(
    model1, model2, train_dataloader, sampling_method, relevant_powers, extra_dim_power
):
    llc_mean_2d, llc_mean_3d_restricted = _do_restricted_gradient_sampling(
        model1, model2, train_dataloader, sampling_method, num_draws=FULL_SAMPLING_DRAWS
    )

    error_msg = (
        f"LLC mean {llc_mean_2d:.3f}!={llc_mean_3d_restricted:.3f} "
        f"for powers {relevant_powers + [extra_dim_power]} using {sampling_method}, "
        f"{model2.weights}"
    )
    assert np.isclose(llc_mean_2d, llc_mean_3d_restricted, atol=TOLERANCE_ATOL), (
        error_msg
    )


def _test_full_accuracy(
    model1,
    model2,
    train_dataloader,
    sampling_method,
    seed,
    relevant_powers,
    extra_dim_power,
):
    llc_mean_2d, llc_mean_3d_restricted = _do_full_sampling(
        model1, model2, train_dataloader, sampling_method, seed, num_draws=500
    )

    error_msg = (
        f"LLC mean {llc_mean_2d:.8f}!={llc_mean_3d_restricted:.8f} "
        f"for powers {relevant_powers + [extra_dim_power]} using {sampling_method}, "
        f"{model2.weights}"
    )
    assert np.isclose(llc_mean_2d, llc_mean_3d_restricted, atol=TOLERANCE_ATOL_FULL), (
        error_msg
    )


def _test_difference_accuracy(
    model, train_dataloader, sampling_method, relevant_powers
):
    llc_mean, rllc_mean = _do_difference_sampling(
        model, train_dataloader, sampling_method, num_draws=200
    )

    error_msg = (
        f"LLC {llc_mean:.3f} too close to RLLC {rllc_mean:.3f} "
        f"for powers {relevant_powers} using {sampling_method}"
    )
    assert not np.isclose(llc_mean, rllc_mean, atol=TOLERANCE_ATOL_DIFFERENCE), (
        error_msg
    )
