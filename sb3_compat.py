import importlib
import sys
import types

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file


def install_numpy_pickle_aliases():
    """Make SB3/cloudpickle models saved with NumPy 2 load under NumPy 1.x.

    Some saved SB3 zip metadata can reference modules such as
    numpy._core.numeric. NumPy 1.x exposes those modules as numpy.core.*.
    Registering aliases before PPO.load lets cloudpickle resolve old paths.
    """

    if "numpy._core" not in sys.modules:
        numpy_core_alias = types.ModuleType("numpy._core")
        numpy_core_alias.__path__ = []
        sys.modules["numpy._core"] = numpy_core_alias
        setattr(np, "_core", numpy_core_alias)
    else:
        numpy_core_alias = sys.modules["numpy._core"]

    module_names = [
        "numeric",
        "multiarray",
        "umath",
        "fromnumeric",
        "shape_base",
        "_multiarray_umath",
    ]

    for module_name in module_names:
        try:
            target_module = importlib.import_module(f"numpy.core.{module_name}")
        except ImportError:
            continue

        alias_name = f"numpy._core.{module_name}"
        sys.modules[alias_name] = target_module
        setattr(numpy_core_alias, module_name, target_module)


def load_ppo_compat(model_path, env, device="auto", ppo_kwargs=None, custom_objects=None, verbose=0):
    """Load an SB3 PPO zip without calling PPO.load.

    On some Windows dependency combinations, PPO.load can terminate with a
    native access violation while the zip metadata and PyTorch weights are
    still readable. Reconstructing PPO explicitly and applying the saved
    parameters avoids that crash while preserving the policy and optimizer
    state.
    """

    install_numpy_pickle_aliases()
    data, params, _pytorch_variables = load_from_zip_file(
        model_path,
        device=device,
        custom_objects=custom_objects,
    )

    kwargs = {
        "learning_rate": 0.0,
        "n_steps": data.get("n_steps", 2048),
        "batch_size": data.get("batch_size", 64),
        "n_epochs": data.get("n_epochs", 10),
        "gamma": data.get("gamma", 0.99),
        "gae_lambda": data.get("gae_lambda", 0.95),
        "clip_range": 0.2,
        "clip_range_vf": data.get("clip_range_vf", None),
        "normalize_advantage": data.get("normalize_advantage", True),
        "ent_coef": data.get("ent_coef", 0.0),
        "vf_coef": data.get("vf_coef", 0.5),
        "max_grad_norm": data.get("max_grad_norm", 0.5),
        "use_sde": data.get("use_sde", False),
        "sde_sample_freq": data.get("sde_sample_freq", -1),
        "target_kl": data.get("target_kl", None),
        "policy_kwargs": data.get("policy_kwargs", {}) or {},
    }
    if ppo_kwargs:
        kwargs.update(ppo_kwargs)

    model = PPO(
        data.get("policy_class", "MlpPolicy"),
        env,
        verbose=verbose,
        device=device,
        **kwargs,
    )
    model.set_parameters(params, exact_match=True, device=device)

    for attr_name in (
        "num_timesteps",
        "_total_timesteps",
        "_num_timesteps_at_start",
        "_n_updates",
        "_episode_num",
        "_current_progress_remaining",
    ):
        if attr_name in data:
            setattr(model, attr_name, data[attr_name])

    return model
