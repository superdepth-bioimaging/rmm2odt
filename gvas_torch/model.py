"""
model.py
========
Forward model for generative virtual angular-scan reconstruction.

The latent variable ``U_stack`` at the centre layer is represented in
polar form as two float32 tensors (``U_stack_abs``, ``U_stack_ang``)
of shape ``(n_angles, Ny, Nx)``. The full-complex field is rebuilt
lazily via :meth:`SynthesisApertureModel.latent_complex`.

Forward pass (given input illumination ``UI_stack`` and a batch of
propagation distances ``z_batch``):

1. Compose ``U_stack = |U| · exp(i · phi) · phasor(UI_stack)``.
   The illumination phasor is re-added here so the subtraction at
   step (5) exactly cancels it; this matches the legacy convention.
2. FFT both stacks once (batched over angles).
3. Build the per-distance angular-spectrum kernel and broadcast
   (B, 1, Ny, Nx) × (1, A, Ny, Nx) → (B, A, Ny, Nx).
4. iFFT back and form ``U_mea_corr = U_mea · conj(phasor(UI_mea))``.
5. Central crop and sum over angles → ``CASS_pred``.

Shapes summary
--------------
UI_stack        (A, Ny, Nx)  complex64
z_batch         (B,)         real   (micrometres)
CASS_pred_batch (B, C, C)    complex64  where C = crop_size
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .physics import (
    angular_spectrum_kernel,
    build_frequency_grid,
    centered_fft2,
    centered_ifft2,
)


class SynthesisApertureModel(nn.Module):
    """Vectorised synthesis-aperture forward model.

    Parameters are stored in polar (magnitude, phase) form with real
    float32 dtypes so that gradients are well-defined and memory is
    halved versus a complex magnitude.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        opt = cfg.optics
        grid = cfg.grid

        self.wavelength: float = opt.wavelength
        self.refractive_index: float = opt.n0
        self.NA: float = opt.NA
        self.scan_step: float = opt.scan_step
        self.k: float = opt.k
        self.npix: float = opt.npix

        self.Nx: int = grid.Nx
        self.Ny: int = grid.Ny
        self.Nz: int = grid.Nz

        crop = grid.resolved_crop()
        self.crop_size: int = crop
        self.crop_start_x: int = (self.Nx - crop) // 2
        self.crop_start_y: int = (self.Ny - crop) // 2
        self.crop_end_x: int = self.crop_start_x + crop
        self.crop_end_y: int = self.crop_start_y + crop

        # Frequency grid as buffers (non-trainable; move with .to(device))
        K2, Kz = build_frequency_grid(self.Nx, self.Ny, self.npix, self.k, device="cpu")
        self.register_buffer("K2", K2)
        self.register_buffer("Kz", Kz)

        # Latent variables
        self.n_angles: int | None = None  # filled on first forward
        self._init_latent_placeholder = True

    # ------------------------------------------------------------------
    # Latent initialisation
    # ------------------------------------------------------------------

    def initialise_latent(self, n_angles: int) -> None:
        """Create learnable parameters for a given number of illumination angles.

        Magnitude is initialised to ones over the full ``(Nx, Ny)`` grid
        (no zero-padded boundary). Both params are complex64 to match legacy
        gradient dynamics (Wirtinger derivatives on both real/imag parts).
        """
        self.n_angles = n_angles

        mag = np.ones((n_angles, self.Nx, self.Ny), dtype=np.complex64)

        self.U_stack_abs = nn.Parameter(
            torch.tensor(mag, dtype=torch.complex64)
        )
        self.U_stack_ang = nn.Parameter(
            torch.zeros((n_angles, self.Ny, self.Nx), dtype=torch.complex64)
        )
        # Put params on whatever device the buffers are on
        dev = self.K2.device
        self.U_stack_abs.data = self.U_stack_abs.data.to(dev)
        self.U_stack_ang.data = self.U_stack_ang.data.to(dev)
        self._init_latent_placeholder = False

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def latent_complex(self) -> torch.Tensor:
        """Return ``|U| · exp(i · phi)`` as complex64 — the current latent field."""
        if self._init_latent_placeholder:
            raise RuntimeError("Call initialise_latent(n_angles) before use.")
        return self.U_stack_abs * torch.exp(1.0j * self.U_stack_ang)

    # Back-compat alias used by the legacy optimisation loop
    def U_stack_(self) -> torch.Tensor:  # noqa: N802
        return self.latent_complex()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        UI_stack: torch.Tensor,  # (A, Ny, Nx) complex64
        z_batch: torch.Tensor,   # (B,) real micrometres
    ) -> torch.Tensor:
        """Predict CASS images for a batch of propagation distances.

        Args:
            UI_stack: illumination field stack.
            z_batch:  physical distances in micrometres (same convention
                      used to build :meth:`build_frequency_grid`).
        Returns:
            ``CASS_pred_batch`` of shape ``(B, crop_size, crop_size)``.
        """
        if self._init_latent_placeholder:
            self.initialise_latent(n_angles=UI_stack.shape[0])

        B = z_batch.shape[0]

        # 1. Compose U_stack, re-adding the illumination phasor.
        # Use exp(1j*angle) instead of safe_phasor: where |UI|=0 (zero-padded
        # border), exp(1j*angle(0))=1 lets the latent field pass through,
        # matching the legacy convention and avoiding hard-edge windowing.
        phasor_UI = torch.exp(1.0j * torch.angle(UI_stack))               # (A, Ny, Nx)
        U_stack = self.latent_complex() * phasor_UI                        # (A, Ny, Nx)

        # 2. Batched FFT over angles.
        UI_spec = centered_fft2(UI_stack)                                  # (A, Ny, Nx)
        U_spec = centered_fft2(U_stack)                                    # (A, Ny, Nx)

        # 3. Propagation kernel for every (B, Ny, Nx).
        z_view = z_batch.to(dtype=torch.complex64).view(B, 1, 1)
        H = angular_spectrum_kernel(z_view, self.K2, self.k, self.Kz)      # (B, Ny, Nx)

        # 4. Broadcast (1,A,Ny,Nx) × (B,1,Ny,Nx) → (B,A,Ny,Nx).
        UI_mea = centered_ifft2(UI_spec.unsqueeze(0) * H.unsqueeze(1))
        U_mea = centered_ifft2(U_spec.unsqueeze(0) * H.unsqueeze(1))

        # 5. Phase-subtract illumination and crop.
        U_mea_corr = U_mea * torch.exp(-1.0j * torch.angle(UI_mea))
        cropped = U_mea_corr[
            :,
            :,
            self.crop_start_y:self.crop_end_y,
            self.crop_start_x:self.crop_end_x,
        ]

        return cropped.sum(dim=1)                                          # (B, C, C)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"Nx={self.Nx}, Ny={self.Ny}, crop={self.crop_size}, "
            f"wl={self.wavelength}, NA={self.NA}, n0={self.refractive_index}, "
            f"n_angles={self.n_angles}"
        )


def build_model(cfg: Config, n_angles: int, device: torch.device | str) -> SynthesisApertureModel:
    """Factory: build on target device with latents pre-initialised."""
    m = SynthesisApertureModel(cfg)
    m.to(device)
    m.initialise_latent(n_angles=n_angles)
    return m
