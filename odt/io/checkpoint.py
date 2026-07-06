"""Persist reconstruction checkpoints to disk. Port of save_checkpoint.m."""
from __future__ import annotations

import os
from typing import Mapping, Optional

import numpy as np
import scipy.io
import torch

from odt.util.backend import to_numpy


def save_checkpoint(
    output_dir: str,
    iter_idx: int,
    x: torch.Tensor,
    *,
    cost: Optional[np.ndarray] = None,
    cost_d: Optional[np.ndarray] = None,
    cost_r: Optional[np.ndarray] = None,
    ang_set: Optional[np.ndarray] = None,
    cfg: Optional[Mapping] = None,
) -> str:
    """Save ``ss_opt_epoch{iter_idx}.mat`` to ``output_dir``.

    Returns the path to the file written.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"ss_opt_epoch{iter_idx}.mat")
    payload = {"ss_opt": to_numpy(x), "iter": iter_idx}
    if cost    is not None: payload["cost"]    = cost
    if cost_d  is not None: payload["cost_d"]  = cost_d
    if cost_r  is not None: payload["cost_r"]  = cost_r
    if ang_set is not None: payload["ang_set"] = ang_set
    # cfg is a python dict — savemat handles dicts as MATLAB structs
    if cfg     is not None: payload["cfg"]     = _flatten_cfg(cfg)
    scipy.io.savemat(file_path, payload, do_compression=True)
    return file_path


def _flatten_cfg(cfg: Mapping) -> dict:
    """Convert nested config dict into a savemat-friendly form."""
    out = {}
    for key, val in cfg.items():
        if isinstance(val, Mapping):
            out[key] = _flatten_cfg(val)
        elif val is None:
            # savemat doesn't like None — skip
            continue
        else:
            out[key] = val
    return out
