import importlib
import sys
import types

import numpy as np


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
