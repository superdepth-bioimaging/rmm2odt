"""
solver.py
=========
Joint Adam optimisation of the centre-layer latent.

Replaces the six duplicated ``Synthesis_to_angular_opt_layers_*.py``
scripts from the legacy codebase with a single, config-driven loop.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from .config import Config
from .data_loading import CASSDataset, load_dataset
from .loss import total_loss, phase_entropy
from .model import SynthesisApertureModel, build_model
from .plotting import plot_loss_curve, plot_U_multi_angles, plot_training_metrics
from . import utils
from .video import save_latent_video
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_output_dir(cfg: Config) -> Path:
    root = Path(cfg.io.output_dir).expanduser()
    name = cfg.io.run_name or utils.default_run_name(cfg)
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_device(cfg: Config) -> torch.device:
    want = cfg.runtime.device
    if want.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(want)


def _denoise_latent(model: torch.nn.Module, kernel_size: int) -> None:
    """Apply spatial median filter to U_stack_abs and U_stack_ang in-place.

    Filters each angle slice independently with a (kernel, kernel) window.
    This acts as a denoising projection step — removes high-frequency noise
    from the latent without biasing the loss function.
    """
    from scipy.ndimage import median_filter

    with torch.no_grad():
        for param in (model.U_stack_abs, model.U_stack_ang):
            arr = param.data.cpu().numpy()
            filtered = median_filter(arr, size=(1, kernel_size, kernel_size))
            param.data.copy_(torch.from_numpy(filtered).to(param.device))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_optimization(cfg: Config) -> Dict[str, Any]:
    """Execute the full optimisation run specified by ``cfg``.

    Returns a dict with keys: ``out_dir``, ``loss_history``,
    ``final_latent`` (numpy complex64 array).
    """
    t0 = time.time()
    _seed_all(cfg.runtime.seed)
    device = _resolve_device(cfg)
    out_dir = _make_output_dir(cfg)

    # ---- Persist the config used for this run (for reproducibility) ----
    with (out_dir / "config_resolved.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2, default=str)

    # ---- Data ----
    UI_stack, CASS, distances = load_dataset(cfg, device)
    n_angles = UI_stack.shape[0]
    log.info(
        "loaded: UI=%s, CASS=%s, n_angles=%d, Z=%d",
        tuple(UI_stack.shape), tuple(CASS.shape), n_angles, distances.shape[0],
    )

    # ---- Model ----
    model = build_model(cfg, n_angles=n_angles, device=device)

    if cfg.runtime.use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log.info("torch.compile enabled (mode=reduce-overhead).")
        except Exception as e:  # noqa: BLE001
            log.warning("torch.compile failed (%s) — continuing without.", e)

    # ---- Dataloader ----
    # Data is already on the target device, so pin_memory must be False
    # and num_workers=0 (workers cannot share CUDA tensors).
    dataset = CASSDataset(CASS, distances)
    loader = DataLoader(
        dataset,
        batch_size=cfg.optimisation.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # ---- Optimiser ----
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.optimisation.learning_rate)
    scheduler: Optional[CosineAnnealingWarmRestarts] = None
    if cfg.optimisation.lr_schedule == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimiser, T_0=max(1, cfg.optimisation.n_epochs // 3)
        )

    # ---- AMP (complex FFTs supported in autocast since PyTorch 2.0) ----
    use_amp = cfg.runtime.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- Loop ----
    loss_history: list[float] = []
    phase_entropy_history: list[float] = []
    n_batches = max(1, len(loader))
    best_loss = float("inf")

    for epoch in range(1, cfg.optimisation.n_epochs + 1):
        model.train()
        epoch_loss = torch.zeros((), device=device)

        for cass_batch, z_batch in loader:
            optimiser.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(UI_stack, z_batch)
                loss = total_loss(
                    pred,
                    cass_batch,
                    latent_complex=_get_latent(model),
                    cfg=cfg,
                )
            # AMP path doesn't yet support full complex autograd reliably;
            # scaler.scale() is a no-op when scaler is disabled.
            scaler.scale(loss).backward()
            if cfg.optimisation.grad_clip_norm > 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optimisation.grad_clip_norm)
            scaler.step(optimiser)
            scaler.update()
            epoch_loss += loss.detach()

        if scheduler is not None:
            scheduler.step()

        mean_loss = (epoch_loss / n_batches).item()
        loss_history.append(mean_loss)

        # Phase entropy metric (not a loss term — just for monitoring)
        with torch.no_grad():
            pe = phase_entropy(_get_latent(model)).item()
        phase_entropy_history.append(pe)

        # Relative improvement of phase entropy vs previous epoch
        if len(phase_entropy_history) >= 2:
            pe_prev = phase_entropy_history[-2]
            pe_rel = (pe_prev - pe) / (abs(pe_prev) + 1e-12)
        else:
            pe_rel = 0.0

        log.info("epoch %3d/%d  loss=%.6e  phase_entropy=%.4f  pe_rel=%.4e",
                 epoch, cfg.optimisation.n_epochs, mean_loss, pe, pe_rel)

        # Save best model
        if mean_loss < best_loss:
            best_loss = mean_loss
            _save_best(model, out_dir, cfg)

        if cfg.optimisation.save_snapshots and (
            epoch % cfg.optimisation.checkpoint_every == 0
            or epoch == cfg.optimisation.n_epochs
        ):
            _save_snapshot(model, out_dir, epoch, loss_history, cfg, phase_entropy_history)

        # Median-filter denoising projection
        dn = cfg.optimisation.denoise_every
        if dn > 0 and epoch % dn == 0:
            _denoise_latent(model, cfg.optimisation.denoise_kernel)


    # ---- Final artefacts ----
    final = _get_latent(model).detach().cpu().numpy()
    save_crop = cfg.optimisation.save_crop_size or cfg.grid.resolved_crop()
    final_cropped = _crop_latent(final, save_crop)
    np.save(out_dir / "U_stack_final.npy", final_cropped)
    np.savetxt(out_dir / "loss_history.txt", np.asarray(loss_history))
    np.savetxt(out_dir / "phase_entropy_history.txt", np.asarray(phase_entropy_history))
    plot_training_metrics(loss_history, phase_entropy_history,
                          out_dir / "training_metrics.png", title=cfg.name)

    # ---- CASS groundtruth vs model prediction (all Z frames) ----
    model.eval()
    bs = cfg.optimisation.batch_size
    pred_chunks = []
    with torch.no_grad():
        for i in range(0, distances.shape[0], bs):
            p = model(UI_stack, distances[i:i + bs])
            pred_chunks.append(p.cpu())
    full_pred = torch.cat(pred_chunks, dim=0).numpy()
    full_gt = CASS.cpu().numpy()
    from .plotting import plot_cass_comparison
    plot_cass_comparison(
        full_gt, full_pred,
        out_dir / "cass_comparison.png",
        distances=distances.real.cpu().numpy(),
        title=f"{cfg.name} — CASS GT vs prediction",
    )

    # ---- Video of all angles ----
    # Normalize each angle by its max amplitude before video
    img_max = np.abs(final_cropped).max(axis=(-2, -1), keepdims=True)
    img_max = np.where(img_max < 1e-12, 1.0, img_max)
    final_normed = final_cropped / img_max
    log.info("rendering latent video (%d angles)...", final_normed.shape[0])
    save_latent_video(final_normed, out_dir / "U_stack_final.mp4",
                      crop_size=None, fps=10)

    # ---- Save ODT input .mat (illumination + U_stack * illumination) ----
    UI_cropped = _crop_latent(UI_stack.cpu().numpy(), save_crop)
    output_stack = final_cropped * UI_cropped
    utils.save_odt_input(UI_cropped, output_stack, out_dir, filename="odt_input.mat")

    elapsed = time.time() - t0
    log.info("done in %.1fs → %s", elapsed, out_dir)

    # ---- Append to results index ----
    results_root = Path(cfg.io.output_dir).expanduser()
    utils.append_results_index(
        results_root, out_dir.name, cfg,
        final_loss=loss_history[-1], elapsed=elapsed,
    )

    return {
        "out_dir": str(out_dir),
        "loss_history": loss_history,
        "final_latent": final_cropped,
    }


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _get_latent(model: torch.nn.Module) -> torch.Tensor:
    """Cope with torch.compile wrapping."""
    if hasattr(model, "latent_complex"):
        return model.latent_complex()
    return model._orig_mod.latent_complex()  # type: ignore[attr-defined]




def _save_best(model: torch.nn.Module, out_dir: Path, cfg: Config) -> None:
    """Overwrite U_stack_best.npy with the current latent (lowest loss so far)."""
    latent = _get_latent(model).detach().cpu().numpy()
    save_crop = cfg.optimisation.save_crop_size or cfg.grid.resolved_crop()
    np.save(out_dir / "U_stack_best.npy", _crop_latent(latent, save_crop))


def _crop_latent(latent: np.ndarray, crop: int) -> np.ndarray:
    """Centre-crop (A, H, W) to (A, crop, crop)."""
    _, H, W = latent.shape
    y0 = (H - crop) // 2
    x0 = (W - crop) // 2
    return latent[:, y0:y0 + crop, x0:x0 + crop]


def _save_snapshot(
    model: torch.nn.Module,
    out_dir: Path,
    epoch: int,
    loss_history: list[float],
    cfg: Config,
    phase_entropy_history: list[float] | None = None,
) -> None:
    latent = _get_latent(model).detach().cpu().numpy()
    save_crop = cfg.optimisation.save_crop_size or cfg.grid.resolved_crop()
    np.save(out_dir / f"U_stack_epoch{epoch:03d}.npy", _crop_latent(latent, save_crop))

    # Loss curve up to this epoch
    plot_loss_curve(loss_history, out_dir / f"loss_epoch{epoch:03d}.png",
                    title=f"{cfg.name} — epoch {epoch}")

    # Dual-axis tracking: loss + phase TV
    if phase_entropy_history:
        plot_training_metrics(loss_history, phase_entropy_history,
                              out_dir / f"metrics_epoch{epoch:03d}.png",
                              title=f"{cfg.name} — epoch {epoch}")

    # Latent field amplitude + phase for selected angles
    angles = cfg.optimisation.snapshot_angles
    if angles:
        valid = [i for i in angles if i < latent.shape[0]]
        if valid:
            plot_U_multi_angles(
                latent, out_dir / f"U_fields_epoch{epoch:03d}.png",
                angle_indices=valid,
                title=f"{cfg.name} — epoch {epoch}",
                crop_size=cfg.optimisation.save_crop_size or cfg.grid.resolved_crop(),
            )
