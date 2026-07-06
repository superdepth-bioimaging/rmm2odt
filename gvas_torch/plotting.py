"""
plotting.py
===========
Static plots for the optimisation run (magnitude, phase, loss curve).

Video assembly lives in :mod:`gvas_torch.video` and is imported only
when needed so that headless runs do not need ffmpeg.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt


def plot_training_metrics(
    loss_history: Sequence[float],
    phase_metric_history: Sequence[float],
    out_path: Path,
    title: str = "",
) -> None:
    """Two-panel plot: (top) loss + phase entropy, (bottom) relative phase entropy improvement.

    Phase entropy rising while loss falls = overfitting.
    Relative improvement going negative = phase is getting worse.
    """
    epochs = np.arange(1, len(loss_history) + 1)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                          gridspec_kw={"height_ratios": [2, 1]})

    # --- Top panel: loss + phase entropy ---
    color_loss = "tab:blue"
    ax_top.plot(epochs, loss_history, marker="o", ms=2, color=color_loss)
    ax_top.set_ylabel("loss", color=color_loss)
    ax_top.tick_params(axis="y", labelcolor=color_loss)
    if all(v > 0 for v in loss_history):
        ax_top.set_yscale("log")

    ax_pe = ax_top.twinx()
    color_pe = "tab:red"
    ax_pe.plot(epochs, phase_metric_history, marker="s", ms=2, color=color_pe)
    ax_pe.set_ylabel("phase entropy", color=color_pe)
    ax_pe.tick_params(axis="y", labelcolor=color_pe)

    if phase_metric_history:
        min_idx = int(np.argmin(phase_metric_history))
        ax_pe.axvline(min_idx + 1, color=color_pe, ls="--", alpha=0.5)
        ax_pe.annotate(f"min @ ep {min_idx + 1}",
                       xy=(min_idx + 1, phase_metric_history[min_idx]),
                       fontsize=8, color=color_pe)

    # --- Bottom panel: relative phase entropy improvement ---
    pe = np.asarray(phase_metric_history)
    if len(pe) >= 2:
        pe_rel = (pe[:-1] - pe[1:]) / (np.abs(pe[:-1]) + 1e-12)
        ax_bot.bar(epochs[1:], pe_rel, width=0.8, color="tab:green", alpha=0.7)
        ax_bot.axhline(0, color="k", ls="-", lw=0.5)
        ax_bot.set_ylabel("rel. phase improvement")
        ax_bot.set_xlabel("epoch")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curve(loss_history: Sequence[float], out_path: Path, title: str = "loss") -> None:
    """Plot epoch-averaged loss versus epoch."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(np.arange(1, len(loss_history) + 1), loss_history, marker="o", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    if all(v > 0 for v in loss_history):
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_U_abs_angle(
    U: np.ndarray,
    out_path: Path,
    angle_idx: int = 0,
    title: str = "",
) -> None:
    """Side-by-side magnitude / phase plot of one slice of ``U`` (shape (A, H, W))."""
    if U.ndim != 3:
        raise ValueError(f"expected 3-D complex array, got shape {U.shape}")
    slc = U[angle_idx]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    im0 = axes[0].imshow(np.abs(slc), cmap="hot")
    axes[0].set_title(f"|U| (angle {angle_idx})")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(np.angle(slc), cmap="jet", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f"arg U (angle {angle_idx})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Centre-crop a 2-D array to (size, size)."""
    h, w = img.shape
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def plot_U_multi_angles(
    U: np.ndarray,
    out_path: Path,
    angle_indices: Sequence[int],
    title: str = "",
    crop_size: int | None = None,
) -> None:
    """Amplitude and phase for multiple angle indices in a grid.

    Layout: top row = amplitudes, bottom row = phases.
    Fields are centre-cropped to ``crop_size`` before plotting.
    """
    if U.ndim != 3:
        raise ValueError(f"expected 3-D complex array, got shape {U.shape}")
    n = len(angle_indices)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 7))
    if n == 1:
        axes = axes[:, np.newaxis]
    for col, idx in enumerate(angle_indices):
        slc = U[idx]
        if crop_size is not None:
            slc = _center_crop(slc, crop_size)
        im0 = axes[0, col].imshow(np.abs(slc), cmap="hot")
        axes[0, col].set_title(f"|U| (ang {idx})")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)
        im1 = axes[1, col].imshow(np.angle(slc), cmap="jet", vmin=-np.pi, vmax=np.pi)
        axes[1, col].set_title(f"phase (ang {idx})")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cass_comparison(
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    distances: np.ndarray | None = None,
    title: str = "",
) -> None:
    """4 rows × Z cols comparison of measured CASS vs forward prediction.

    Rows: |GT|, |Pred|, |GT - Pred|, phase residual ``angle(Pred · conj(GT))``.
    All frames in ``gt`` are shown (one column per z-slice).

    Args:
        gt:        (Z, H, W) complex — measured CASS target
        pred:      (Z, H, W) complex — model forward prediction
        distances: (Z,) physical depth offsets, used for column titles (optional)
    """
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError(f"shape mismatch: gt={gt.shape}, pred={pred.shape}")
    Z = gt.shape[0]

    abs_gt = np.abs(gt)
    abs_pred = np.abs(pred)
    abs_diff = np.abs(gt - pred)
    phase_gt = np.angle(gt)
    phase_pred = np.angle(pred)
    phase_diff = np.angle(pred * np.conj(gt))

    amp_vmax = float(max(abs_gt.max(), abs_pred.max()))
    diff_vmax = float(abs_diff.max()) if abs_diff.max() > 0 else 1.0

    fig, axes = plt.subplots(6, Z, figsize=(2.0 * Z + 1.5, 12.5))
    if Z == 1:
        axes = axes[:, np.newaxis]

    rows = [
        (abs_gt,    "|GT|",       "hot",      0.0,      amp_vmax),
        (abs_pred,  "|Pred|",     "hot",      0.0,      amp_vmax),
        (abs_diff,  "|GT-Pred|",  "hot",      0.0,      diff_vmax),
        (phase_gt,  "∠GT",        "twilight", -np.pi,   np.pi),
        (phase_pred,"∠Pred",      "twilight", -np.pi,   np.pi),
        (phase_diff,"Δφ",         "twilight", -np.pi,   np.pi),
    ]
    last_im = None
    for r, (data, label, cmap, vmin, vmax) in enumerate(rows):
        for c in range(Z):
            ax = axes[r, c]
            last_im = ax.imshow(data[c], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            if r == 0:
                if distances is not None:
                    ax.set_title(f"d={float(distances[c]):+.1f}", fontsize=8)
                else:
                    ax.set_title(f"z={c}", fontsize=8)
        plt.colorbar(last_im, ax=axes[r, -1], fraction=0.046)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_epoch_snapshot(
    U: np.ndarray,
    out_dir: Path,
    epoch: int,
    angle_idx: int = 0,
) -> None:
    """Save per-epoch |U| and arg U to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_U_abs_angle(
        U,
        out_dir / f"U_abs_angle_epoch{epoch:03d}.png",
        angle_idx=angle_idx,
        title=f"epoch {epoch}",
    )
