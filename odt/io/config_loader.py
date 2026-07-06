"""Resolve a config by name (string) or pass through a dict.

Port of ``resolve_config.m``. In Python the configs are functions in
the ``configs/`` package that return a dict.
"""
from __future__ import annotations

import importlib
import math
from typing import Any, Mapping, Union


def resolve_config(name_or_dict: Union[str, Mapping[str, Any]]) -> dict:
    """Load a config by name string or pass a dict through unchanged.

    Parameters
    ----------
    name_or_dict : str | mapping
        ``str``  -- name of a config function in the ``configs`` package,
                    e.g. ``'ht29_cell_515nm'``. The function is called and
                    its returned dict is used.
        mapping  -- returned (deep-copied) unchanged.

    Returns
    -------
    dict
        Validated dict with derived fields filled in (``physics.k0``,
        ``physics.k`` if absent).
    """
    if isinstance(name_or_dict, Mapping):
        cfg = _to_dict(name_or_dict)
    elif isinstance(name_or_dict, str):
        try:
            module = importlib.import_module(f"configs.{name_or_dict}")
        except ModuleNotFoundError as e:
            raise FileNotFoundError(
                f"Config module 'configs.{name_or_dict}' not found. "
                f"Make sure 'configs/' is on sys.path."
            ) from e
        if not hasattr(module, name_or_dict):
            raise AttributeError(
                f"Module 'configs.{name_or_dict}' must define a function "
                f"named '{name_or_dict}' that returns a config dict."
            )
        cfg = module.__dict__[name_or_dict]()
        if not isinstance(cfg, dict):
            raise TypeError(
                f"Config function '{name_or_dict}' returned "
                f"{type(cfg).__name__}; expected dict."
            )
    else:
        raise TypeError(
            "Input must be a dict or a config-name string; "
            f"got {type(name_or_dict).__name__}"
        )

    # Fill derived physics fields if absent
    if "physics" in cfg:
        if "k0" not in cfg["physics"]:
            cfg["physics"]["k0"] = 2.0 * math.pi / cfg["physics"]["wl"]
        if "k" not in cfg["physics"]:
            cfg["physics"]["k"] = cfg["physics"]["k0"] * cfg["physics"]["n0"]

    return cfg


def _to_dict(cfg: Mapping) -> dict:
    """Recursively convert a (possibly mapping-typed) config to a plain dict."""
    out = {}
    for k, v in cfg.items():
        if isinstance(v, Mapping):
            out[k] = _to_dict(v)
        else:
            out[k] = v
    return out
