"""Proximal operator of the 3D Total Variation regularizer.

Port of ``denoise_tv_3d.m`` / ``denoiseTV_3D_gpu.m``. Solves::

    x = argmin_x  0.5 * ||x - y||^2  +  lambda * TV(x)

via Fast Gradient Projection ('fgp', default) or plain Gradient
Projection ('gp'). Algorithm preserved exactly from the original
(Joowon Lim, EPFL).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def denoise_tv_3d(
    y: torch.Tensor,
    lambda_: float,
    *,
    maxiter: int = 100,
    L: float = 12.0,
    tol: float = 1e-4,
    optim: str = "fgp",
    verbose: bool = False,
    bounds: Tuple[float, float] = (float("-inf"), float("inf")),
    P_4D_init: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    """Total-Variation proximal operator on the GPU.

    Parameters
    ----------
    y : torch.Tensor, (Ny, Nx, Nz), real
        Input volume to be denoised.
    lambda_ : float
        TV weight.
    maxiter : int
        Maximum outer iterations.
    L : float
        Lipschitz constant of the dual problem (default 12).
    tol : float
        Early-stop tolerance on relative dual change.
    optim : {'fgp', 'gp'}
        Outer optimisation method.
    verbose : bool
        Print per-iter relative-difference if True.
    bounds : (float, float)
        Box constraint ``[lb, ub]`` applied to the primal x at each step.
    P_4D_init : torch.Tensor, (Ny, Nx, Nz, 3), real, optional
        Warm-start dual variable.

    Returns
    -------
    x : torch.Tensor, (Ny, Nx, Nz), real
        Denoised volume.
    P_4D : torch.Tensor, (Ny, Nx, Nz, 3), real
        Final dual variable.
    iters : int
        Number of iterations performed.
    L : float
        Lipschitz constant used (for chained calls).
    """
    if y.is_complex():
        # Original code assumes real y; if complex was passed, take real part.
        # (Matches MATLAB behavior where P_4D is real.)
        y_real = y.real.contiguous()
    else:
        y_real = y

    sz = y_real.shape  # (Ny, Nx, Nz)
    device = y_real.device
    dtype = y_real.dtype

    if P_4D_init is None:
        P_4D = torch.zeros((*sz, 3), dtype=dtype, device=device)
    else:
        P_4D = P_4D_init.to(device=device, dtype=dtype).clone()

    if verbose:
        print("******************************************")
        print("**   3D Denoising with TV Regularizer   **")
        print("******************************************")
        print("#iter     relative-dif")
        print("======================")

    count = 0
    flag = False
    re = float("inf")
    iters = maxiter

    if optim == "fgp":
        t = 1.0
        F_4D = P_4D.clone()
        for i in range(maxiter):
            TMP = _adj_tv_op_3d(F_4D)
            K = y_real - lambda_ * TMP

            TMP = _project_box(K, bounds)
            TMP_4D_1 = _tv_op_3d(TMP)

            TMP_4D_2 = F_4D + (1.0 / (L * lambda_)) * TMP_4D_1
            Pnew_4D = _project_l2(TMP_4D_2)

            re = (
                torch.linalg.norm(Pnew_4D - P_4D).item()
                / max(torch.linalg.norm(Pnew_4D).item(), 1e-30)
            )
            count = count + 1 if re < tol else 0

            tnew = (1.0 + math.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
            F_4D = Pnew_4D + (t - 1.0) / tnew * (Pnew_4D - P_4D)
            P_4D = Pnew_4D
            t = tnew

            if verbose:
                print(f"{i + 1:3d} \t {re:10.5f}")

            if count >= 5:
                flag = True
                iters = i + 1
                break

    elif optim == "gp":
        for i in range(maxiter):
            TMP = _adj_tv_op_3d(P_4D)
            K = y_real - lambda_ * TMP

            TMP = _project_box(K, bounds)
            TMP_4D_1 = _tv_op_3d(TMP)

            TMP_4D_2 = P_4D + (1.0 / (L * lambda_)) * TMP_4D_1
            Pnew_4D = _project_l2(TMP_4D_2)

            re = (
                torch.linalg.norm(Pnew_4D - P_4D).item()
                / max(torch.linalg.norm(Pnew_4D).item(), 1e-30)
            )
            count = count + 1 if re < tol else 0
            P_4D = Pnew_4D

            if verbose:
                print(f"{i + 1:3d} \t {re:10.5f}")

            if count >= 5:
                flag = True
                iters = i + 1
                break

    else:
        raise ValueError(f"Unknown optimisation method: {optim!r}")

    # Final primal update
    TMP = _adj_tv_op_3d(P_4D)
    K = y_real - lambda_ * TMP
    x = _project_box(K, bounds)

    return x, P_4D, iters, L


# ----------------------------------------------------------------------
# Helpers — exact ports of the nested MATLAB functions in denoiseTV_3D_gpu.m
# ----------------------------------------------------------------------

def _tv_op_3d(f: torch.Tensor) -> torch.Tensor:
    """Forward TV operator with circular (=reflexive in this code) BC.

    Shape: (Ny, Nx, Nz) -> (Ny, Nx, Nz, 3).
    """
    Df = torch.empty((*f.shape, 3), dtype=f.dtype, device=f.device)
    # MATLAB:  shift(1:end-1,:,:) = f(2:end,:,:); shift(end,:,:) = f(1,:,:); Df(:,:,:,1) = shift - f;
    Df[..., 0] = torch.roll(f, shifts=-1, dims=0) - f
    Df[..., 1] = torch.roll(f, shifts=-1, dims=1) - f
    Df[..., 2] = torch.roll(f, shifts=-1, dims=2) - f
    return Df


def _adj_tv_op_3d(P: torch.Tensor) -> torch.Tensor:
    """Adjoint TV operator. Shape: (Ny, Nx, Nz, 3) -> (Ny, Nx, Nz).

    Equivalent to ``g[k] = (P[k-1] - P[k])`` summed across the 3 axes,
    with circular boundary at k=0 (i.e. P[-1] = P[N-1]). This is exactly
    the negative divergence on the dual variable.
    """
    dy = torch.roll(P[..., 0], shifts=1, dims=0) - P[..., 0]
    dx = torch.roll(P[..., 1], shifts=1, dims=1) - P[..., 1]
    dz = torch.roll(P[..., 2], shifts=1, dims=2) - P[..., 2]
    return dy + dx + dz


def _project_l2(B: torch.Tensor) -> torch.Tensor:
    """Project per-voxel gradient vector onto the unit L2 ball.

    Shape: (Ny, Nx, Nz, 3) -> same.
    """
    norms = torch.sqrt(torch.sum(B ** 2, dim=3, keepdim=True))
    return B / torch.clamp(norms, min=1.0)


def _project_box(x: torch.Tensor, bounds: Tuple[float, float]) -> torch.Tensor:
    """Box-constraint projection with -inf / +inf supported."""
    lb, ub = bounds
    if lb == float("-inf") and ub == float("inf"):
        return x
    if lb == float("-inf"):
        return torch.clamp(x, max=ub)
    if ub == float("inf"):
        return torch.clamp(x, min=lb)
    return torch.clamp(x, min=lb, max=ub)
