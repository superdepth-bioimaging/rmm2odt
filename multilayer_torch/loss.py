"""
multilayer_torch/loss.py
-------------------------
Loss and regularisation functions for the multi-layer scattering model.

All functions are pure PyTorch — no numpy, no TF.

Direct ports of bioimaging_subfunctions:
    pearson_loss          ← Pearson_correlation_2
    l2_loss               ← L2_norm
    tv_loss               ← total_variation_3D
    layers_correlation_loss ← layers_correlation

New (PyTorch-only):
    entropy_loss          — Shannon entropy of sample layer intensity
                            (minimising → sparser, higher-contrast reconstruction)
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _covariance(a: torch.Tensor, b: torch.Tensor, dim) -> torch.Tensor:
    """Absolute covariance |∑ (a − ā)·conj(b − b̄)| reduced over *dim*.

    Matches the covariance() helper inside Pearson_correlation_2 exactly.

    Parameters
    ----------
    a, b : complex Tensor  — must have the same shape
    dim  : int or tuple    — spatial dimensions to reduce over

    Returns
    -------
    real Tensor  — one covariance value per element in the non-reduced dims
    """
    a_c = a - a.mean(dim=dim, keepdim=True)
    b_c = b - b.mean(dim=dim, keepdim=True)
    return torch.abs((a_c * b_c.conj()).sum(dim=dim))


def _centre_crop(t: torch.Tensor, size: int) -> torch.Tensor:
    """Centre-crop last two spatial dims of *t* to *size* × *size*."""
    h, w = t.shape[-2], t.shape[-1]
    r0 = h // 2 - size // 2
    c0 = w // 2 - size // 2
    return t[..., r0:r0 + size, c0:c0 + size]


def _to_complex(param: torch.Tensor) -> torch.Tensor:
    """Convert a 2-channel [2, H, W] float32 parameter to complex64 [H, W]."""
    return torch.complex(param[0], param[1])


# ---------------------------------------------------------------------------
# Pearson correlation loss  (main optimization objective)
# ---------------------------------------------------------------------------

def pearson_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Negative Pearson correlation loss.

    Ports Pearson_correlation_2 from bioimaging_subfunctions.py exactly.

    Physics / motivation
    --------------------
    We minimise negative correlation rather than MSE so the loss is invariant
    to global amplitude scaling — important when the absolute transmission
    strength is unknown.

    Parameters
    ----------
    y_true, y_pred : complex64 Tensor [B, H, W]
        B = batch size; H, W = spatial dimensions.

    Returns
    -------
    scalar Tensor  — negative sum of per-batch Pearson correlations.
    """
    dim = (-2, -1)   # reduce over spatial dims, keep batch dim
    cov_yp = _covariance(y_true, y_pred, dim=dim)                  # [B]
    cov_yy = _covariance(y_true, y_true, dim=dim)                  # [B]
    cov_pp = _covariance(y_pred, y_pred, dim=dim)                  # [B]
    pearson = cov_yp / torch.sqrt(cov_yy * cov_pp + 1e-12)
    return -pearson.sum()


def mse_loss_norm(y_true: torch.Tensor, y_pred: torch.Tensor,
                  eps: float = 1e-12) -> torch.Tensor:
    """Per-pattern energy-normalised MSE loss for complex64 [B, H, W] tensors.

    Each batch element is rescaled to unit L2 energy independently before the
    squared-error sum. Result is amplitude-invariant per pattern but
    phase-sensitive — unlike Pearson's `|cov|/...`, a global phase mismatch
    between y_pred and y_true contributes to the loss.

    Returns a scalar Tensor (sum over batch + spatial).
    """
    dim = (-2, -1)
    energy_t = (y_true.conj() * y_true).real.sum(dim=dim, keepdim=True)
    energy_p = (y_pred.conj() * y_pred).real.sum(dim=dim, keepdim=True)
    yt = y_true / torch.sqrt(energy_t + eps)
    yp = y_pred / torch.sqrt(energy_p + eps)
    diff = yp - yt
    return (diff.conj() * diff).real.sum()


# ---------------------------------------------------------------------------
# L2 regularisation on layer amplitudes
# ---------------------------------------------------------------------------

def l2_loss(layers: list, sizes: list) -> torch.Tensor:
    """L2 regularisation on layer amplitudes.

    Ports L2_norm from bioimaging_subfunctions.py.

    Formula (matching tf.nn.l2_loss convention):
        l2_reg = (1/N) ∑_k  [∑ |layer_k|² / 2] / size_k²

    Parameters
    ----------
    layers : list of complex64 Tensor [H_k, W_k]
        Layer parameter tensors (already converted to complex, no batch dim).
    sizes  : list of int
        Spatial size of each layer (size_k = H_k = W_k).

    Returns
    -------
    scalar Tensor
    """
    n = len(layers)
    if n == 0:
        return torch.zeros((), dtype=torch.float32)
    total = sum(
        0.5 * (layer.abs() ** 2).sum() / (size ** 2)
        for layer, size in zip(layers, sizes)
    )
    return total / n


# ---------------------------------------------------------------------------
# Lateral TV regularisation (per-layer spatial smoothness)
# ---------------------------------------------------------------------------

def _tv_2d(img: torch.Tensor) -> torch.Tensor:
    """2-D total variation on a [H, W] or [..., H, W] complex tensor.

    Uses tf.nn.l2_loss convention: sum(|diff|²) / 2 for each direction.
    """
    dh = (img[..., 1:, :] - img[..., :-1, :]).abs()
    dw = (img[..., :, 1:] - img[..., :, :-1]).abs()
    return 0.5 * (dh ** 2).sum() + 0.5 * (dw ** 2).sum()


def lateral_tv(layers: list, sizes: list) -> torch.Tensor:
    """Per-layer 2-D Total Variation regularisation.

    Penalises spatial gradients within each layer independently.
    No axial (inter-layer) term — layers treated as independent surfaces.

    Formula:
        tv = (1/N) * sum_k  TV_2D(layer_k) / size_min^2

    Parameters
    ----------
    layers : list of complex64 Tensor [H_k, W_k]
    sizes  : list of int

    Returns
    -------
    scalar Tensor
    """
    n = len(layers)
    if n == 0:
        return torch.zeros((), dtype=torch.float32)
    size_min = min(sizes)
    cropped = [_centre_crop(layer, size_min) for layer in layers]
    return sum(_tv_2d(c) / (size_min ** 2) for c in cropped) / n


# ---------------------------------------------------------------------------
# Depth TV regularisation (inter-layer phase smoothness)
# ---------------------------------------------------------------------------

def depth_tv_selected(
    layers: list,
    sizes: list,
    layer_indices: list,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Depth-wise phase smoothness between selected adjacent layers.

    Normalises each layer to unit phasors (layer / |layer|), then penalises
    |phasor_a - phasor_b|^2 between adjacent pairs. This measures pure phase
    difference while ignoring amplitude, and is numerically stable.

    Physics / motivation
    --------------------
    In a scattering medium, the phase of the transmission function is expected
    to vary smoothly across depth (gradual refractive index changes). This
    regularisation enforces that prior without constraining amplitude, which
    may change abruptly at layer boundaries.

    Parameters
    ----------
    layers : list of complex64 Tensor [H_k, W_k]
    sizes  : list of int
    layer_indices : list of int
        Sorted indices of layers to regularise between.
    eps : float
        Stability constant for division by amplitude.

    Returns
    -------
    scalar Tensor
    """
    if len(layer_indices) < 2:
        return torch.tensor(0.0, device=layers[0].device)
    size_min = min(int(sizes[i]) for i in layer_indices)
    phasors = {}
    for i in layer_indices:
        c = _centre_crop(layers[i], size_min)
        phasors[i] = c / (torch.abs(c) + eps)
    dl = sum(
        torch.mean(torch.abs(phasors[a] - phasors[b]) ** 2)
        for a, b in zip(layer_indices[:-1], layer_indices[1:])
    )
    return dl / (len(layer_indices) - 1)


# ---------------------------------------------------------------------------
# Layer-to-layer correlation regularisation
# ---------------------------------------------------------------------------

def layers_correlation_loss(layers: list, sizes: list) -> torch.Tensor:
    """Pearson correlation regularisation across layers (lateral + axial).

    Ports layers_correlation from bioimaging_subfunctions.py.

    Lateral component: Pearson between adjacent rows AND adjacent columns of
        each layer — penalises high spatial correlation (encourages smooth
        but distinct per-layer features).
    Axial component:   Pearson between every pair of adjacent layers
        (after cropping to size_min).

    Formula:
        lateral = ∑_k [Pearson(layer_k[1:,:], layer_k[:-1,:])
                      + Pearson(layer_k[:,1:], layer_k[:,:-1])]
        axial   = ∑_{k} Pearson(layer_{k+1}, layer_k)
        reg     = (axial + lateral) / (3 · N)

    Parameters
    ----------
    layers : list of complex64 Tensor [H_k, W_k]
    sizes  : list of int

    Returns
    -------
    scalar Tensor  (negative, because pearson_loss is negative correlation)
    """
    n = len(layers)
    size_min = min(sizes)
    cropped = [_centre_crop(layer, size_min) for layer in layers]

    # Lateral: treat rows of [H, W] as batch [H, W] — reduce over last dim only.
    # Pearson between row i and row i-1 for each layer.
    def _pearson_rows(img: torch.Tensor) -> torch.Tensor:
        a, b = img[1:, :], img[:-1, :]   # [H-1, W]
        return _pearson_1d(a, b)

    def _pearson_cols(img: torch.Tensor) -> torch.Tensor:
        a, b = img[:, 1:], img[:, :-1]   # [H, W-1]
        return _pearson_1d(a, b)

    correlation_lateral = sum(
        _pearson_rows(c) + _pearson_cols(c) for c in cropped
    )
    correlation_axial = sum(
        _pearson_2d(c1, c0)
        for c0, c1 in zip(cropped[:-1], cropped[1:])
    )
    return (correlation_axial + correlation_lateral) / (3.0 * n)


def _pearson_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Negative Pearson correlation, reducing over the last dim only.

    Inputs [*, L] — each row/col pair is an independent sample.
    Returns scalar (sum over all samples).
    """
    cov_ab = _covariance(a, b, dim=-1)
    cov_aa = _covariance(a, a, dim=-1)
    cov_bb = _covariance(b, b, dim=-1)
    pearson = cov_ab / torch.sqrt(cov_aa * cov_bb + 1e-12)
    return -pearson.sum()


def _pearson_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Negative Pearson correlation over last two dims, returns scalar."""
    cov_ab = _covariance(a, b, dim=(-2, -1))
    cov_aa = _covariance(a, a, dim=(-2, -1))
    cov_bb = _covariance(b, b, dim=(-2, -1))
    pearson = cov_ab / torch.sqrt(cov_aa * cov_bb + 1e-12)
    return -pearson.sum()


# ---------------------------------------------------------------------------
# Model-B coherent intensity loss (sample-plane Wiener-corrected energy)
# ---------------------------------------------------------------------------

def coherent_intensity_loss(
    E_in_batch: torch.Tensor,
    E_back_batch: torch.Tensor,
    crop_size: int,
    eps_alpha: float = 0.02,
) -> torch.Tensor:
    """Coherent-sum total-intensity cost (Model B).

    Per speckle realisation k, the corrected confocal image is the
    Wiener division of the back-propagated measurement by the
    illumination field at the sample plane:

        I_k(r) = E_back_k(r) · conj(E_in_k(r)) / (|E_in_k(r)|² + ε_k)

    where ε_k = α · mean_r |E_in_k(r)|²  (per-realisation, adaptive).

    The loss is the negative coherent total intensity over realisations:

        S(r) = Σ_k I_k(r)
        loss = − Σ_r |S(r)|²

    Coherent buildup gives Σ_r |Σ_k I_k|² ∝ N² when the scattering
    correction is consistent across speckles, vs ∝ N when it is not —
    the same N² vs N signal that drives CLASS / power-iteration
    aberration estimation. Minimising this loss drives the shared
    pupil + scattering-layer parameters toward correction estimates
    that produce a speckle-invariant sample image.

    Parameters
    ----------
    E_in_batch : complex64 Tensor [B, H, W]
        Illumination field at the sample plane, one per speckle realisation.
    E_back_batch : complex64 Tensor [B, H, W]
        Back-propagated measurement at the sample plane, one per realisation.
        Must have same shape as E_in_batch.
    crop_size : int
        Centre-crop size for the spatial sum (matches physical FoV).
    eps_alpha : float
        Wiener regularisation fraction; ε_k = α · mean_r |E_in_k|².

    Returns
    -------
    scalar Tensor — negative coherent total intensity (minimise to maximise it).
    """
    # Per-realisation Wiener division: I_k = E_back · conj(E_in) / (|E_in|² + ε_k)
    # ε_k = α · mean_r |E_in_k(r)|².
    in_abs_sq = E_in_batch.abs() ** 2                      # [B, H, W]
    eps_k = eps_alpha * in_abs_sq.mean(dim=(-2, -1), keepdim=True)
    I_k = E_back_batch * E_in_batch.conj() / (in_abs_sq + eps_k)

    # Coherent sum across realisations, then total intensity over the
    # cropped physical FoV.
    S = I_k.sum(dim=0)                                     # [H, W]
    S = _centre_crop(S, crop_size)
    return -(S.abs() ** 2).sum()


# ---------------------------------------------------------------------------
# Shannon entropy regularisation on the sample layer
# ---------------------------------------------------------------------------

def entropy_loss(sample_layer: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the sample layer intensity.

    Minimising entropy encourages sparse, high-contrast reconstructions —
    pixels concentrate intensity into a few bright features rather than
    spreading it uniformly.

    Physics / motivation
    --------------------
    For sparse samples (neurons, fluorescent beads, sub-cellular organelles)
    the true object has high contrast and low entropy.  Adding this term to
    the loss biases the optimiser towards that prior.

    WARNING — not appropriate for dense, diffuse samples (thick tissue,
    collagen matrix) where the ground-truth itself has high entropy.

    Formula
    -------
        p_i  = |E_i|² / (∑_j |E_j|² + ε)        intensity probability
        H    = −∑_i p_i · log(p_i)                Shannon entropy

    Minimising H pushes the distribution p towards a delta function
    (one bright pixel) rather than a uniform distribution (maximum entropy).

    Parameters
    ----------
    sample_layer : complex64 Tensor [H, W]
        The reconstructed sample plane — last output layer, no batch dim.

    Returns
    -------
    scalar Tensor  (positive; minimise to reduce entropy)
    """
    intensity = sample_layer.abs() ** 2
    p = intensity / (intensity.sum() + 1e-12)
    p = p.clamp(min=1e-10)           # avoid log(0)
    return -(p * torch.log(p)).sum()
