"""
multilayer_torch/analysis.py
-----------------------------
Image quality metrics for the reconstructed sample object and pupil aberrations.

Two usage modes
---------------
1. During optimization (called from solver._save_epoch_plots every epoch):
       hf  = compute_hf_power(sample_layer_complex)
       sbr = compute_sbr(sample_layer_complex)
       snr_in  = compute_pupil_snr(pupil_in_complex)
       snr_out = compute_pupil_snr(pupil_out_complex)
   All four histories are passed to the figure functions.

2. Post-hoc analysis of previously saved checkpoints:
       python cli.py analyze --output-dir PATH --pixel-size 0.7 --n-layers 6

Metrics
-------
HF power fraction  (complex field PSD)
    Fraction of total spectral power in the high-frequency band of the
    complex sample layer (freq > cutoff_fraction x Nyquist).
    The complex field is normalized by max amplitude before FFT so that
    amplitude growth across epochs does not inflate the metric.
    Peaks at the optimal reconstruction epoch, then declines as overfitting
    degrades spatial resolution.

SBR  (Signal-to-Background Ratio)
    99th-percentile / median of the amplitude map.
    Scale-invariant — amplitude ratio is unaffected by global amplitude growth.
    Measures contrast between object and background.

Pupil amplitude SNR
    Measures smoothness of the pupil aberration amplitude map.
    Computed as: mean(smooth) / std(amp - smooth), where smooth is a
    Gaussian-blurred version of the amplitude.
    Why SNR instead of PSD HF fraction:
      The granular noise in an overfitting pupil is pixel-scale (1-3 px).
      Its power is spread across a very wide high-frequency band, so the
      azimuthally-averaged PSD value in any single frequency bin is tiny
      and the HF fraction metric is insensitive.
      SNR directly measures the noise floor relative to the signal level —
      even a small amount of granularity causes std(residual) to rise and
      SNR to drop noticeably.
    High SNR = smooth pupil (good). As overfitting starts, input pupil SNR
    drops while output pupil SNR stays high, revealing the divergence.

Optimal stopping epoch
    Intersection of two criteria:
      - HF power of sample layer peaks  (best reconstruction quality)
      - Input pupil SNR begins to drop below output pupil SNR  (overfitting onset)
    The epoch just before both conditions simultaneously worsen is optimal.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper: radial PSD from a 2-D real or complex array
# ---------------------------------------------------------------------------

def _radial_psd(arr2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Azimuthally averaged, total-normalized radial PSD.

    arr2d may be real or complex; the caller is responsible for any
    normalization before calling (e.g. dividing by max amplitude).

    Returns
    -------
    bin_index : integer radial bin indices 0..num_bins-1
    psd_curve : normalized PSD (sums to 1), same length as bin_index
    """
    H, W = arr2d.shape
    psd_2d = np.abs(np.fft.fftshift(np.fft.fft2(arr2d))) ** 2

    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[:H, :W]
    r_int = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)

    num_bins = min(cy, cx)
    psd_curve = np.zeros(num_bins, dtype=np.float64)
    for ir in range(num_bins):
        mask = r_int == ir
        if mask.any():
            psd_curve[ir] = psd_2d[mask].mean()

    total = psd_curve.sum()
    if total > 1e-30:
        psd_curve /= total

    return np.arange(num_bins), psd_curve


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

def compute_radial_psd(
    img_complex: np.ndarray,
    pixel_size_um: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Azimuthally averaged, normalized radial PSD of a 2-D complex image.

    The complex field is divided by max(|img|) before FFT so that amplitude
    is scaled to [0, 1] while the phase is preserved intact. The PSD
    therefore captures both amplitude and phase structure of the wavefield,
    independent of global amplitude scale.

    Parameters
    ----------
    img_complex   : complex ndarray [H, W]
    pixel_size_um : spatial sampling interval in um/pixel

    Returns
    -------
    freq_axis : 1-D ndarray, spatial frequencies in 1/um
    psd_curve : 1-D ndarray, normalized PSD (sums to 1)
    """
    amp_max = np.abs(img_complex).max()
    H, W = img_complex.shape
    num_bins = min(H, W) // 2
    if amp_max < 1e-12:
        freq_axis = np.arange(num_bins) / (min(H, W) * pixel_size_um)
        return freq_axis, np.zeros(num_bins)

    img_norm = img_complex.astype(np.complex128) / amp_max
    _, psd_curve = _radial_psd(img_norm)
    freq_axis = np.arange(len(psd_curve)) / (min(H, W) * pixel_size_um)
    return freq_axis, psd_curve


def compute_hf_power(
    img_complex: np.ndarray,
    pixel_size_um: float = 1.0,
    cutoff_fraction: float = 0.25,
) -> float:
    """
    HF power fraction of the complex field (amplitude + phase).

    Fraction of total spectral power above cutoff_fraction x Nyquist.
    Applied to the sample layer to track reconstruction quality per epoch.
    Peaks at the optimal epoch, then declines as overfitting degrades resolution.

    Returns scalar in [0, 1].
    """
    freq_axis, psd_curve = compute_radial_psd(img_complex, pixel_size_um)
    nyquist = 1.0 / (2.0 * pixel_size_um)
    return float(psd_curve[freq_axis >= cutoff_fraction * nyquist].sum())


def compute_pupil_snr(
    pupil_complex: np.ndarray,
    smooth_sigma: float = 3.0,
) -> float:
    """
    SNR of the pupil aberration amplitude map.

    Computed as:
        smooth = gaussian_filter(|pupil|, sigma=smooth_sigma)
        noise  = |pupil| - smooth
        SNR    = mean(smooth) / std(noise)

    Why SNR instead of PSD HF fraction for pupil noise?
    -----------------------------------------------------
    Overfitting noise in the pupil amplitude is pixel-scale granularity
    (1-3 px). Its power is spread across a very wide high-frequency band
    in the PSD, so the azimuthally-averaged value per bin is tiny and the
    HF fraction metric is insensitive (values near 0.0001 even with visible
    granularity). SNR directly measures the noise floor: even a small amount
    of granularity raises std(noise) noticeably, causing SNR to drop sharply.

    A physical pupil has a smooth amplitude (slowly varying apodization).
    High SNR = smooth, well-behaved pupil.
    Dropping SNR = granular noise = overfitting onset.

    Why Gaussian residual?
    -----------------------
    Gaussian blur with sigma=3 px removes structure coarser than ~3 px.
    The residual therefore contains only pixel-scale noise — exactly the
    granularity we want to detect. Legitimate aberration features (which
    vary smoothly over tens of pixels) are captured by the smooth component
    and do not contribute to std(noise).

    Parameters
    ----------
    pupil_complex : complex ndarray [H, W]
    smooth_sigma  : Gaussian blur radius in pixels (default 3)

    Returns
    -------
    SNR scalar. Higher = smoother = less noise.
    """
    from scipy.ndimage import gaussian_filter
    amp = np.abs(pupil_complex).astype(np.float64)
    if amp.max() < 1e-12:
        return 0.0
    smooth = gaussian_filter(amp, sigma=smooth_sigma)
    noise  = amp - smooth
    signal_level = smooth.mean()
    noise_std    = noise.std()
    return float(signal_level / (noise_std + 1e-12))


def compute_sbr(img_complex: np.ndarray, crop_size: int | None = None) -> float:
    """
    Signal-to-Background Ratio: 99th-percentile / median amplitude.

    Scale-invariant — amplitude ratio cancels the global scale factor.
    Higher SBR = brighter object relative to background.

    Parameters
    ----------
    img_complex : complex [H, W] array
    crop_size : int, optional
        If provided, SBR is computed only on the central `crop_size x crop_size`
        region. Necessary for padded sample layers where the outer region is
        dark (light cone mask) — including those zeros in the median inflates
        SBR to meaningless values.
    """
    if crop_size is not None:
        h, w = img_complex.shape[-2], img_complex.shape[-1]
        r0 = h // 2 - crop_size // 2
        c0 = w // 2 - crop_size // 2
        img_complex = img_complex[..., r0:r0 + crop_size, c0:c0 + crop_size]
    amp = np.abs(img_complex).astype(np.float64)
    p99 = float(np.percentile(amp, 99))
    p50 = float(np.median(amp))
    return p99 / (p50 + 1e-12)


def compute_entropy(img_complex: np.ndarray) -> float:
    """
    Shannon entropy of the sample layer intensity distribution.

    Formula
    -------
        p_i  = |E_i|^2 / sum_j |E_j|^2        intensity probability
        H    = -sum_i p_i * log(p_i)            Shannon entropy (nats)

    Interpretation as a quality metric
    -----------------------------------
    Low entropy  -> intensity concentrated in few bright pixels
                 -> sparse, high-contrast reconstruction (good for beads/neurons)
    High entropy -> intensity spread uniformly across all pixels
                 -> blurry or noisy reconstruction

    During optimization, entropy tends to DECREASE as the reconstruction sharpens.
    Entropy is complementary to SBR: SBR measures amplitude ratio, entropy
    measures the overall sparsity of the intensity distribution.

    WARNING: entropy is only a meaningful quality metric for sparse samples.
    Dense tissue has legitimately high entropy in the ground truth.

    Parameters
    ----------
    img_complex : complex ndarray [H, W]

    Returns
    -------
    scalar (nats). Lower = sparser = sharper reconstruction.
    """
    intensity = np.abs(img_complex).astype(np.float64) ** 2
    total = intensity.sum()
    if total < 1e-30:
        return 0.0
    p = intensity / total
    p = np.where(p > 1e-10, p, 1e-10)
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _ylim(data: list[float], pad: float = 0.12) -> tuple[float, float]:
    """Return (lo, hi) y-limits with *pad* fractional padding on each side.

    Gives the data room to breathe without anchoring the axis at zero, so
    small epoch-to-epoch variations fill the full axis height.

    If all values are identical, falls back to ±10 % of the value (or ±1
    if the value is zero).
    """
    arr = [v for v in data if v is not None and np.isfinite(v)]
    if not arr:
        return (0.0, 1.0)
    lo, hi = min(arr), max(arr)
    rng = hi - lo
    if rng < 1e-12:
        margin = abs(lo) * 0.10 if abs(lo) > 1e-12 else 1.0
        return (lo - margin, hi + margin)
    return (lo - pad * rng, hi + pad * rng)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_quality_tracking_figure(
    epochs: list[int],
    hf_history: list[float],
    sbr_history: list[float],
    output_dir: str | Path,
    entropy_history: Optional[list[float]] = None,
) -> None:
    """
    Quality tracking figure. Overwrites quality_metrics.png on every call.

    If entropy_history is None (or empty):
        Single panel — HF power fraction (left/blue) and SBR (right/green).

    If entropy_history is provided:
        Two panels stacked vertically:
          Top    — HF power (left/blue) and SBR (right/green)
          Bottom — Shannon entropy (nats), lower = sparser = better

    Entropy decreases as the reconstruction sharpens, so a falling entropy
    curve confirms that the optimisation is increasing sparsity.
    """
    if not epochs or not hf_history:
        return

    color_hf      = '#1f77b4'   # blue
    color_sbr     = '#2ca02c'   # green
    color_entropy = '#9467bd'   # purple

    has_entropy = bool(entropy_history)
    nrows = 2 if has_entropy else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(7, 4 * nrows),
                             squeeze=False)
    ax1 = axes[0, 0]

    # --- Top panel: HF power + SBR ---
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('HF power fraction (sample layer)', color=color_hf)
    ax1.plot(epochs, hf_history, color=color_hf, linewidth=2,
             marker='o', markersize=4, label='HF power')
    ax1.tick_params(axis='y', labelcolor=color_hf)
    ax1.set_ylim(*_ylim(hf_history))

    ax2 = ax1.twinx()
    ax2.set_ylabel('SBR', color=color_sbr)
    if sbr_history:
        ax2.plot(epochs[:len(sbr_history)], sbr_history,
                 color=color_sbr, linewidth=2, linestyle='--',
                 marker='s', markersize=4, label='SBR')
        ax2.set_ylim(*_ylim(sbr_history))
    ax2.tick_params(axis='y', labelcolor=color_sbr)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    ax1.set_title('Reconstruction quality metrics', fontsize=12)

    # --- Bottom panel: Shannon entropy ---
    if has_entropy:
        ax3 = axes[1, 0]
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Shannon entropy (nats)', color=color_entropy)
        ax3.plot(epochs[:len(entropy_history)], entropy_history,
                 color=color_entropy, linewidth=2, marker='D', markersize=4,
                 label='Entropy')
        ax3.tick_params(axis='y', labelcolor=color_entropy)
        ax3.set_ylim(*_ylim(entropy_history))
        ax3.legend(loc='upper right', fontsize=9)
        ax3.set_title('Sample layer Shannon entropy (lower = sparser)', fontsize=11)

    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'quality_metrics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_overfitting_figure(
    epochs: list[int],
    hf_history: list[float],
    pupil_snr_in: list[float],
    pupil_snr_out: list[float],
    output_dir: str | Path,
) -> None:
    """
    Dual-axis overfitting diagnostic figure.

    Left axis  : input pupil amplitude SNR (red solid)
                 output pupil amplitude SNR (orange dashed)
    Right axis : HF power of sample layer (blue dotted, for epoch alignment)

    How to read it
    --------------
    SNR is higher = smoother = less noise (opposite direction to a noise metric).
    Early epochs: SNR_in ~ SNR_out, both high (smooth pupils), HF power rising.
    Overfitting onset: SNR_in starts to DROP below SNR_out while HF power declines.
    Optimal stopping epoch: last epoch before SNR_in begins its sustained drop.
    """
    if not epochs or not hf_history or not pupil_snr_in:
        return

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color_in  = '#d62728'   # red
    color_out = '#ff7f0e'   # orange
    color_hf  = '#1f77b4'   # blue

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Pupil amplitude SNR', color=color_in)
    ax1.plot(epochs[:len(pupil_snr_in)], pupil_snr_in,
             color=color_in, linewidth=2, marker='o', markersize=4,
             label='Pupil SNR (input)')
    if pupil_snr_out:
        ax1.plot(epochs[:len(pupil_snr_out)], pupil_snr_out,
                 color=color_out, linewidth=2, linestyle='--',
                 marker='s', markersize=4, label='Pupil SNR (output)')
    ax1.tick_params(axis='y', labelcolor=color_in)
    # Span both SNR curves jointly so their relative divergence is visible
    all_snr = list(pupil_snr_in) + (list(pupil_snr_out) if pupil_snr_out else [])
    ax1.set_ylim(*_ylim(all_snr))

    ax2 = ax1.twinx()
    ax2.set_ylabel('HF power fraction (sample layer)', color=color_hf)
    ax2.plot(epochs[:len(hf_history)], hf_history,
             color=color_hf, linewidth=1.5, linestyle=':',
             marker='^', markersize=4, label='HF power (ref)')
    ax2.tick_params(axis='y', labelcolor=color_hf)
    ax2.set_ylim(*_ylim(hf_history))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    ax1.set_title('Overfitting diagnostic: pupil noise vs reconstruction quality',
                  fontsize=11)
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'overfitting.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_psd_comparison_figure(
    images_dict: dict[str, np.ndarray],
    output_path: str | Path,
    pixel_size_um: float = 1.0,
) -> None:
    """
    Normalized radial PSD curves for multiple images on the same axes.

    Parameters
    ----------
    images_dict   : dict mapping label -> complex [H, W] ndarray
    output_path   : full file path for the saved PNG
    pixel_size_um : um/pixel (same for all images)
    """
    if not images_dict:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    n = len(images_dict)
    colors = [plt.cm.viridis(i / max(n - 1, 1)) for i in range(n)]
    nyquist = 1.0 / (2.0 * pixel_size_um)

    for (label, img), color in zip(images_dict.items(), colors):
        freq_axis, psd_curve = compute_radial_psd(img, pixel_size_um)
        ax.plot(freq_axis, psd_curve, color=color, linewidth=1.5, label=label)

    ax.axvline(nyquist, color='gray', linestyle=':', linewidth=1, label='Nyquist')
    ax.set_xlabel('Spatial frequency (1/um)')
    ax.set_ylabel('Normalized PSD')
    ax.set_title('Radial PSD comparison')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Post-hoc checkpoint analysis
# ---------------------------------------------------------------------------

def analyze_from_checkpoints(
    output_dir: str | Path,
    pixel_size_um: float,
    n_layers: int,
    epochs: Optional[list[int]] = None,
    cutoff_fraction: float = 0.25,
    psd_epochs: Optional[list[int]] = None,
    save_dir: Optional[str | Path] = None,
) -> dict:
    """
    Post-hoc quality analysis from saved .txt checkpoint files.

    Loads for each epoch:
      - output_layer_{n_layers}_epoch_{e}.txt  -> HF power, SBR
      - pupil_aberration_1_epoch_{e}.txt       -> input pupil amplitude noise
      - pupil_aberration_2_epoch_{e}.txt       -> output pupil amplitude noise

    Saves:
      - quality_metrics.png    HF power + SBR tracking
      - overfitting.png        pupil noise (input vs output) + HF power reference
      - psd_comparison.png     radial PSD curves for selected epochs
      - quality_metrics.npz    all arrays for downstream use

    Parameters
    ----------
    output_dir      : directory containing checkpoint .txt files
    pixel_size_um   : um/pixel of the sample layer
    n_layers        : total number of output layers (sample = output_layer_{n_layers})
    epochs          : 1-based epoch indices to analyse; None -> auto-discover
    cutoff_fraction : HF cutoff as fraction of Nyquist (default 0.25)
    psd_epochs      : subset of epochs for PSD comparison; None -> auto-select up to 8
    save_dir        : directory to write plots and .npz; defaults to output_dir

    Returns
    -------
    dict with keys 'epochs', 'hf_power', 'sbr', 'entropy',
                   'pupil_snr_in', 'pupil_snr_out'
    """
    output_dir = Path(output_dir)
    save_dir = Path(save_dir) if save_dir is not None else output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover epochs from sample layer checkpoint files
    if epochs is None:
        pattern = f'output_layer_{n_layers}_epoch_*.txt'
        found = sorted(output_dir.glob(pattern))
        if not found:
            raise FileNotFoundError(
                f'No files matching {pattern} found in {output_dir}'
            )
        discovered = []
        for f in found:
            parts = f.stem.rsplit('_epoch_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                discovered.append(int(parts[1]))
        epochs = sorted(discovered)
        logger.info(f'Auto-discovered epochs: {epochs}')

    hf_history: list[float] = []
    sbr_history: list[float] = []
    entropy_history: list[float] = []
    pupil_snr_in: list[float] = []
    pupil_snr_out: list[float] = []
    epoch_list: list[int] = []
    psd_images: dict[str, np.ndarray] = {}

    for e in epochs:
        # --- Sample layer ---
        fpath = output_dir / f'output_layer_{n_layers}_epoch_{e}.txt'
        if not fpath.exists():
            logger.warning(f'Sample layer not found, skipping epoch {e}: {fpath.name}')
            continue
        try:
            layer = np.loadtxt(str(fpath), dtype=np.complex64)
        except Exception as exc:
            logger.warning(f'Failed to load {fpath.name}: {exc}')
            continue

        hf  = compute_hf_power(layer, pixel_size_um, cutoff_fraction)
        sbr = compute_sbr(layer)
        ent = compute_entropy(layer)
        hf_history.append(hf)
        sbr_history.append(sbr)
        entropy_history.append(ent)
        epoch_list.append(e)
        psd_images[f'epoch {e}'] = layer

        # --- Pupil amplitude SNR ---
        snr_in, snr_out = 0.0, 0.0
        for idx, dest in [(1, 'in'), (2, 'out')]:
            fp = output_dir / f'pupil_aberration_{idx}_epoch_{e}.txt'
            if fp.exists():
                try:
                    pupil = np.loadtxt(str(fp), dtype=np.complex64)
                    val = compute_pupil_snr(pupil)
                    if dest == 'in':
                        snr_in = val
                    else:
                        snr_out = val
                except Exception as exc:
                    logger.warning(f'Failed to load {fp.name}: {exc}')
        pupil_snr_in.append(snr_in)
        pupil_snr_out.append(snr_out)

        logger.info(
            f'  epoch {e:3d}  HF={hf:.4f}  SBR={sbr:.2f}'
            f'  entropy={ent:.2f}'
            f'  pupil_SNR in={snr_in:.1f}  out={snr_out:.1f}'
        )

    if not epoch_list:
        logger.warning('No valid checkpoints loaded -- analysis skipped.')
        return {}

    # --- Figures ---
    save_quality_tracking_figure(epoch_list, hf_history, sbr_history, save_dir,
                                 entropy_history=entropy_history)
    logger.info(f'Saved quality_metrics.png -> {save_dir}')

    save_overfitting_figure(epoch_list, hf_history, pupil_snr_in, pupil_snr_out,
                            save_dir)
    logger.info(f'Saved overfitting.png -> {save_dir}')

    # PSD comparison: select up to 8 epochs for readability
    if psd_epochs is not None:
        psd_subset = {k: v for k, v in psd_images.items()
                      if int(k.split()[-1]) in psd_epochs}
    elif len(psd_images) > 8:
        keys = list(psd_images.keys())
        step = max(1, len(keys) // 7)
        selected = keys[::step]
        if keys[-1] not in selected:
            selected.append(keys[-1])
        psd_subset = {k: psd_images[k] for k in selected}
    else:
        psd_subset = psd_images

    save_psd_comparison_figure(psd_subset, save_dir / 'psd_comparison.png', pixel_size_um)
    logger.info(f'Saved psd_comparison.png -> {save_dir}')

    # --- NPZ archive ---
    np.savez(
        save_dir / 'quality_metrics.npz',
        epochs=np.array(epoch_list),
        hf_power=np.array(hf_history),
        sbr=np.array(sbr_history),
        entropy=np.array(entropy_history),
        pupil_snr_in=np.array(pupil_snr_in),
        pupil_snr_out=np.array(pupil_snr_out),
    )
    logger.info(f'Saved quality_metrics.npz -> {save_dir}')

    # --- Report best epoch ---
    _report_best_epoch(epoch_list, hf_history, pupil_snr_in, pupil_snr_out)

    return {
        'epochs': epoch_list,
        'hf_power': hf_history,
        'sbr': sbr_history,
        'entropy': entropy_history,
        'pupil_snr_in': pupil_snr_in,
        'pupil_snr_out': pupil_snr_out,
    }


def _report_best_epoch(
    epochs: list[int],
    hf_history: list[float],
    pupil_snr_in: list[float],
    pupil_snr_out: list[float],
) -> None:
    """
    Log the recommended stopping epoch based on combined HF power + pupil SNR.

    Scoring:
        score = norm(HF_power) - norm(max(SNR_out - SNR_in, 0))

    HF_power peaks at optimal reconstruction quality.
    When overfitting starts, SNR_in drops below SNR_out — the gap
    (SNR_out - SNR_in) is the overfitting penalty term.
    The epoch maximising the score balances peak quality against
    minimal overfitting.
    """
    if not epochs:
        return

    hf_arr  = np.array(hf_history,   dtype=float)
    snr_in  = np.array(pupil_snr_in,  dtype=float)
    snr_out = np.array(pupil_snr_out, dtype=float)
    # penalty: only when input SNR drops below output SNR (overfitting)
    diverge = np.maximum(snr_out - snr_in, 0.0)

    def _norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)

    score = _norm(hf_arr) - _norm(diverge)
    best_idx = int(np.argmax(score))

    logger.info(
        f'Recommended stopping epoch: {epochs[best_idx]}'
        f'  HF={hf_history[best_idx]:.4f}'
        f'  pupil_SNR_in={pupil_snr_in[best_idx]:.1f}'
        f'  pupil_SNR_out={pupil_snr_out[best_idx]:.1f}'
        f'  score={score[best_idx]:.4f}'
    )
    logger.info(f'  (Best HF power alone: epoch {epochs[int(np.argmax(hf_arr))]})')
