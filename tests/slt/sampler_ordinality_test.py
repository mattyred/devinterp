import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from devinterp.optim import SGLD, SGMCMC
from devinterp.slt.llc import LLCEstimator
from devinterp.slt.sampler import sample
from devinterp.utils import *
from torch.utils.data import DataLoader, TensorDataset


@pytest.mark.parametrize("sampling_method", [SGLD, SGMCMC.sgld])
@pytest.mark.parametrize("model_name", ["Polynomial", "LinePlusDot"])
@pytest.mark.parametrize("dim", [2, 10])
def test_linedot_normal_crossing(
    generated_linedot_normalcrossing_dataset, sampling_method, model_name, dim, request
):
    seed = 42
    torch.manual_seed(seed)
    Model = request.getfixturevalue(model_name)
    if model_name == "Polynomial":
        model = Model([2 for _ in range(dim)])
    else:
        model = Model(dim)
    train_dataloader, _, _, _ = generated_linedot_normalcrossing_dataset
    lr = (
        0.0001 / dim
    )  # to account for smaller steps in higher D. might not work well for SGNHT?
    num_chains = 5
    num_draws = 1_000
    llcs = []
    sample_points = [
        [0.0 for _ in range(dim)],
        [0.0 if i == dim - 1 else 1.0 for i in range(dim)],
    ]
    for sample_point in sample_points:
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
            num_chains=num_chains,
            num_draws=num_draws,
            callbacks=[llc_estimator],
            verbose=False,
        )
        llcs += [llc_estimator.get_results()["llc/mean"]]
    assert (
        np.diff(llcs) >= 0
    ).all(), f"Ordinality not preserved for sampler {sampling_method} on {dim}-d {model_name}: llcs {llcs} are not in ascending order."
