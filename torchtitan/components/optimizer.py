# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Generic, Iterator, TypeVar
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from torchtitan.components.ft import FTManager, has_torchft
from torchtitan.config_manager import JobConfig
from torchtitan.distributed.parallel_dims import ParallelDims

__all__ = [
    "OptimizersContainer",
    "MuonOptimizersContainer",
    "OptimizersContainerWithDecayGroups",
    "build_optimizers",
]


if has_torchft:
    import torchft as ft


T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(Optimizer, Stateful, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    optimizers: list[T]
    model_parts: list[nn.Module]

    def _no_weight_decay(self, name, param) -> bool:
        is_no_weight_decay = any([k in name for k in self._no_weight_decay_keys])
        if is_no_weight_decay and param.ndim > 1:
            print(f"\033[93mWarning: Parameter '{name}' with shape={param.shape} is excluded from weight decay. Typically, only 1D parameters are expected be excluded.\033[0m")

        return is_no_weight_decay

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        no_weight_decay_keys: list[str] | None = None,
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        self._no_weight_decay_keys = no_weight_decay_keys or []

        if not self._no_weight_decay_keys:
            for model in self.model_parts:
                params = [p for p in model.parameters() if p.requires_grad]
                self.optimizers.append(optimizer_cls(params, **optimizer_kwargs))
                all_params.extend(params)
        else:
            for model in self.model_parts:
                # 分成两组
                decay_params = []
                no_decay_params = []
                no_wd_names = []
                for name, param in model.named_parameters():
                    if self._no_weight_decay(name, param):
                        no_decay_params.append(param)
                        no_wd_names.append(name)
                    else:
                        decay_params.append(param)

                if decay_params:
                    all_params.append({
                        "params": decay_params,
                        "weight_decay": optimizer_kwargs["weight_decay"],
                    })
                if no_decay_params:
                    all_params.append({
                        "params": no_decay_params,
                        "weight_decay": 0.0,  # ← 关键：这些参数不做 weight decay
                    })

                # 移除全局 kwargs 中的 weight_decay，因为已经在 all_params 中指定了
                kwargs_without_wd = {
                    k: v for k, v in optimizer_kwargs.items() if k != "weight_decay"
                }
                print(f"\033[92mParameters excluded from weight decay(number={len(no_wd_names)}): {no_wd_names}\033[0m")
            self.optimizers.append(optimizer_cls(all_params, **kwargs_without_wd))
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)


def _is_lm_head_param(name: str, param: nn.Parameter) -> bool:
    """Determine if a parameter is the language model head (output projection)."""
    name_parts = name.split(".")
    return "output" in name_parts and param.ndim == 2


def _is_muon_param(name: str, param: nn.Parameter) -> bool:
    """Determine if a parameter should use the Muon algorithm.

    Muon's Newton-Schulz orthogonalization is only applicable to 2D weight
    matrices. Embeddings and the output projection (lm_head) use AdamW instead.
    """
    if param.ndim != 2:
        return False
    if "tok_embeddings" in name or "embed" in name:
        return False
    if _is_lm_head_param(name, param):
        return False
    return True


class MuonOptimizersContainer(OptimizersContainer):
    """OptimizersContainer for the Muon optimizer from Microsoft's dion library.

    Muon uses a single optimizer instance with multiple param groups that
    have different algorithms: 'muon' for 2D weight matrices and 'adamw'
    (or 'lion') for everything else (embeddings, biases, layernorms, lm_head).
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_kwargs: dict[str, Any],
        fsdp_mesh: "ParallelDims",
        fallback_algorithm: str = "adamw",
    ) -> None:
        try:
            from dion import Muon
        except ImportError:
            raise ImportError(
                "Muon optimizer requires the 'dion' package. "
                "Install it with: pip install git+https://github.com/microsoft/dion.git"
            )

        from torchtitan.tools.logging import logger

        all_params = []
        self.model_parts = model_parts

        muon_params = []
        fallback_params = []

        for model in self.model_parts:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                all_params.append(param)
                if _is_muon_param(name, param):
                    muon_params.append(param)
                else:
                    fallback_params.append(param)

        logger.info(
            f"Muon optimizer: {len(muon_params)} params with 'muon' algorithm, "
            f"{len(fallback_params)} params with '{fallback_algorithm}' algorithm"
        )

        param_groups = []
        if muon_params:
            param_groups.append({"params": muon_params, "algorithm": "muon"})
        if fallback_params:
            param_groups.append(
                {"params": fallback_params, "algorithm": fallback_algorithm}
            )

        # fsdp_mesh = parallel_dims

        muon_optimizer = Muon(
            param_groups,
            distributed_mesh=fsdp_mesh,
            **optimizer_kwargs,
        )

        self.optimizers = [muon_optimizer]
        self._post_init(all_params, optimizer_kwargs)

    def _validate_length(self, expected_length: int) -> None:
        pass

    # pyrefly: ignore [bad-override]
    def step(self, *args, **kwargs) -> None:
        self.optimizers[0].step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        self.optimizers[0].zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in (func(model, self.optimizers[0]) for model in self.model_parts)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        for model in self.model_parts:
            func(model, self.optimizers[0])


class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.model_parts = model_parts

        optim_dict = {}
        for model in self.model_parts:
            for p in model.parameters():
                if p.requires_grad:
                    optim_dict[p] = optimizer_cls([p], **optimizer_kwargs)
                all_params.append(p)

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())

        self._validate_length(
            sum(len(list(model.parameters())) for model in self.model_parts)
        )
        self._post_init(all_params, optimizer_kwargs)

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass


class FTOptimizersContainer(OptimizersContainer):
    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        ft_manager: "ft.Manager",
        use_ft_optimizer: bool = True,
    ) -> None:
        super().__init__(model_parts, optimizer_cls, optimizer_kwargs)

        # Force to initialize the optimizer state so that `optim.step()`
        # won't be called by state_dict() and load_state_dict().
        _ = {
            k: v
            for sd in map(get_optimizer_state_dict, model_parts, self.optimizers)
            for k, v in sd.items()
        }
        self.cache_state_dict: dict[str, Any] = {}
        self._ft_optimizer = ft.Optimizer(ft_manager, self)
        # Whether to determine quorum using FT.optimizer,
        # in semi-sync training we use the synchronization step to start quorum
        self._use_ft_optimizer: bool = use_ft_optimizer

    def init_cache_state_dict(self) -> None:
        self.cache_state_dict = super().state_dict()

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # We have to invalidate the `cache_state_dict` because optimizer uses
        # assign instead of copy when doing `load_state_dict()`. Without
        # invalidating the `cache_state_dict`, there will be memory leakage.
        self.cache_state_dict = {}
        super().load_state_dict(state_dict)
        self.init_cache_state_dict()

    def step(self, *args, **kwargs) -> None:
        """Calling the correct step() depending on the caller.

        TorchFT's OptimizerWrapper.step() is designed to be called only once
        per train step per ft.Manager regardless how many optimizers are used.
        Hence we will need to appropriately dispatch the call.
        """
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.step(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        """Calling the correct zero_grad() depending on the caller.

        Check the comment in ``step()``.
        """
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.zero_grad(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().zero_grad(*args, **kwargs)


def build_optimizers(
    model_parts: list[nn.Module],
    job_config: JobConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager,
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``job_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        job_config (JobConfig): Job config containing the optimizer name and parameters.
    """
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    if optim_in_bwd and job_config.parallelism.pipeline_parallel_degree > 1:
        raise NotImplementedError(
            "Optimizers in backward is not supported with pipeline parallelism."
        )
    name = job_config.optimizer.name

    if name == "Muon":
        if optim_in_bwd:
            raise NotImplementedError(
                "Optimizer in backward is not supported with Muon."
            )
        if ft_manager and ft_manager.enabled:
            raise NotImplementedError(
                "TorchFT is not supported with Muon optimizer."
            )

        adjust_lr = job_config.optimizer.adjust_lr
        optimizer_kwargs = {
            "lr": job_config.optimizer.lr,
            "mu": job_config.optimizer.mu,
            "betas": (job_config.optimizer.beta1, job_config.optimizer.beta2),
            "weight_decay": job_config.optimizer.weight_decay,
            "epsilon": job_config.optimizer.eps,
            "adjust_lr": adjust_lr if adjust_lr != "none" else None,
        }

        return MuonOptimizersContainer(
            model_parts=model_parts,
            optimizer_kwargs=optimizer_kwargs,
            fsdp_mesh=parallel_dims.build_mesh("cuda")["dp_shard_cp"],
            # parallel_dims=parallel_dims,
        )

    lr = job_config.optimizer.lr
    beta1 = job_config.optimizer.beta1
    beta2 = job_config.optimizer.beta2
    eps = job_config.optimizer.eps
    weight_decay = job_config.optimizer.weight_decay

    optim_implementation = job_config.optimizer.implementation
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"

    no_weight_decay_keys = job_config.optimizer.no_weight_decay_keys
    optimizer_kwargs = {
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": eps,
        "weight_decay": weight_decay,
        "fused": fused,
        "foreach": foreach,
    }

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]

    if optim_in_bwd and ft_manager.enabled:
        raise ValueError("TorchFT is not supported with optimizers in backward.")
    elif optim_in_bwd:
        assert not no_weight_decay_keys, f"no_weight_decay_keys not support in this branch"
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )
    elif ft_manager.enabled:
        assert not no_weight_decay_keys, f"no_weight_decay_keys not support in this branch"
        return FTOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager.manager,
            use_ft_optimizer=job_config.fault_tolerance.semi_sync_method is None,
        )
    else:
        return OptimizersContainer(
            model_parts, optimizer_cls, optimizer_kwargs,
            no_weight_decay_keys=no_weight_decay_keys,
        )
