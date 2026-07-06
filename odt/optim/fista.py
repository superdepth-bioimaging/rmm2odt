"""FISTA Nesterov momentum step. Port of fista_step.m."""
from __future__ import annotations

import math
from typing import Tuple

import torch


def fista_step(
    x_curr: torch.Tensor,
    x_prev: torch.Tensor,
    q_old: float,
) -> Tuple[torch.Tensor, float]:
    """Nesterov momentum update for FISTA.

    Implements the standard FISTA acceleration::

        q_new      = 0.5 * (1 + sqrt(1 + 4 * q_old**2))
        x_momentum = x_curr + ((q_old - 1) / q_new) * (x_curr - x_prev)

    Parameters
    ----------
    x_curr : torch.Tensor
        Current proximal-step result.
    x_prev : torch.Tensor
        Previous iterate.
    q_old : float
        Previous momentum scalar.

    Returns
    -------
    x_momentum : torch.Tensor
        Extrapolated point at which the next gradient is evaluated.
    q_new : float
        Updated momentum scalar.

    Notes
    -----
    Pure function (no side effects). Trivial to unit-test.
    """
    q_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * q_old ** 2))
    x_momentum = x_curr + ((q_old - 1.0) / q_new) * (x_curr - x_prev)
    return x_momentum, q_new
