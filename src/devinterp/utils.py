import os
import warnings
from functools import partial
from itertools import islice
from typing import Any, Callable, Dict, Mapping, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

PJRT_DEVICE = os.environ.get("PJRT_DEVICE", None)
USE_TPU_BACKEND = os.environ.get(
    "USE_TPU_BACKEND", True if (PJRT_DEVICE == "TPU") else False
)
TPU_TYPE = os.environ.get("TPU_TYPE", None)


class Outputs(NamedTuple):
    loss: torch.Tensor
    # Add more outputs here if needed


EvalResults = Union[
    Outputs, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor
]

EvaluateFn = Callable[[nn.Module, torch.Tensor], EvalResults]


def plot_trace(
    trace,
    y_axis,
    x_axis="step",
    title=None,
    plot_mean=True,
    plot_std=True,
    fig_size=(12, 9),
    true_lc=None
):
    import matplotlib.pyplot as plt

    num_chains, num_draws = trace.shape
    sgld_step = list(range(num_draws))
    if true_lc:
        plt.axhline(y=true_lc, color="r", linestyle="dashed")
    # trace
    for i in range(num_chains):
        draws = trace[i]
        plt.plot(sgld_step, draws, linewidth=1, label=f"chain {i}")

    # mean
    mean = np.mean(trace, axis=0)
    plt.plot(
        sgld_step,
        mean,
        color="black",
        linestyle="--",
        linewidth=2,
        label="mean",
        zorder=3,
    )

    # std
    std = np.std(trace, axis=0)
    plt.fill_between(
        sgld_step, mean - std, mean + std, color="gray", alpha=0.3, zorder=2
    )

    if title is None:
        title = f"{y_axis} values over sampling draws"
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.figure(figsize=fig_size)
    plt.tight_layout()
    plt.show()


def default_nbeta(
    dataloader: Union[DataLoader, int], gradient_accumulation_steps: int = 1
) -> float:
    if isinstance(dataloader, DataLoader):
        default_nbeta = dataloader.batch_size * gradient_accumulation_steps
        if default_nbeta <= 1:
            warnings.warn(
                "default nbeta is undefined for batch_size * gradient_accumulation_steps == 1, falling back to default value of 1"
            )
            return 1
        else:
            return default_nbeta / np.log(default_nbeta)
    elif isinstance(dataloader, int):
        default_nbeta = dataloader * gradient_accumulation_steps
        if default_nbeta <= 1:
            warnings.warn(
                "default nbeta is undefined for batch_size * gradient_accumulation_steps == 1, falling back to default value of 1"
            )
            return 1
        else:
            return default_nbeta / np.log(default_nbeta)
    else:
        raise NotImplementedError(
            f"N*beta for data type {type(dataloader)} not implemented, use DataLoader or int instead."
        )


def prepare_input(
    data: Union[torch.Tensor, Any], device, is_deepspeed_enabled=False, accelerator=None
) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.

    Adapted from HuggingFace's transformers's Trainer._prepare_input().
    """
    if isinstance(data, Mapping):
        return type(data)(
            {
                k: prepare_input(
                    v,
                    device=device,
                    is_deepspeed_enabled=is_deepspeed_enabled,
                    accelerator=accelerator,
                )
                for k, v in data.items()
            }
        )
    elif isinstance(data, (tuple, list)):
        return type(data)(
            prepare_input(
                v,
                device=device,
                is_deepspeed_enabled=is_deepspeed_enabled,
                accelerator=accelerator,
            )
            for v in data
        )
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        if is_deepspeed_enabled and (
            torch.is_floating_point(data) or torch.is_complex(data)
        ):
            # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            # embedding. Other models such as wav2vec2's inputs are already float and thus
            # may need special handling to match the dtypes of the model
            kwargs.update(
                {"dtype": accelerator.state.deepspeed_plugin.hf_ds_config.dtype()}
            )
        return data.to(**kwargs)
    return data


def split_results(results: EvalResults) -> Tuple[torch.Tensor, Any]:
    if isinstance(results, dict):
        loss = results.pop("loss")
    elif isinstance(results, tuple):
        loss = results[0]
        if len(results) > 1:
            results = results[1:]
    elif isinstance(results, torch.Tensor):
        loss = results
        results = None
    elif hasattr(results, "loss"):
        loss = results.loss
    else:
        raise ValueError("compute_loss must return a dict, tuple, or torch.Tensor")

    return loss, results


def get_seeded_dataloader(dataloader, seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataloader.dataset, batch_size=dataloader.batch_size, generator=gen
    )


def get_init_loss_one_batch(dataloader, model, evaluate: EvaluateFn, device, seed=None):
    if seed is not None:
        dataloader = get_seeded_dataloader(dataloader, seed)
    model = model.to(device)
    model.train()  # to make sure we're using train loss, comparable to train loss of sampler()

    with torch.no_grad():
        data = next(iter(dataloader))
        data = prepare_input(data, device=device)
        loss = split_results(evaluate(model, data))[0].detach().item()

    return loss


def get_init_loss_multi_batch(
    dataloader, n_batches, model, evaluate, device, seed=None
):
    if seed is not None:
        dataloader = get_seeded_dataloader(dataloader, seed)

    model = model.to(device)
    model.train()
    loss = 0.0
    n_batches = min(n_batches, len(dataloader))

    with torch.no_grad():
        for data in islice(dataloader, n_batches):
            data = prepare_input(data, device=device)
            loss += split_results(evaluate(model, data))[0].detach().item()

    return loss / n_batches


def get_init_loss_full_batch(dataloader, model, evaluate, device, seed=None):
    if seed is not None:
        dataloader = get_seeded_dataloader(dataloader, seed)

    model = model.to(device)
    model.train()
    loss = 0.0

    with torch.no_grad():
        for data in dataloader:
            data = prepare_input(data, device=device)
            loss += split_results(evaluate(model, data))[0].detach().item()

    return loss / len(dataloader)


def evaluate(criterion, model, data):
    x, y = data
    y_pred = model(x)
    return criterion(y_pred, y), {"output": y_pred}


def make_evaluate(criterion):
    return partial(evaluate, criterion)


evaluate_ce = partial(evaluate, F.cross_entropy)
evaluate_mse = partial(evaluate, F.mse_loss)


def set_seed(seed: int, device: Optional[Union[str, torch.device]] = None):
    """
    Sets the seed for the Learner.

    Args:
        seed (int): Seed to set.
    """
    import random

    # np.random.SeedSequence outputs np.uint32, but random.seed() expects int
    seed = int(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch_xla.core.xla_model as xm

        if device is None:
            xm.set_rng_state(seed)
        elif "xla" in str(device):
            xm.set_rng_state(seed, device=device)

    except (ImportError, RuntimeError):
        pass

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cycle(iterable, limit=None):
    """
    Use this function to cycle through a dataloader. Unlike itertools.cycle, this function doesn't cache
    values in memory.

    Note: Be careful with cycling a shuffled interable. The shuffling will be different for each loop dependent on the seed
    state, unlike with itertools.cycle.

    :param iterable: Iterable to cycle through
    :param limit: Number of cycles to go through. If None, cycles indefinitely.
    """
    index = 0
    if limit is None:
        limit = float("inf")
    while True:
        for x in iterable:
            if index >= limit:
                return
            else:
                yield x
            index += 1
