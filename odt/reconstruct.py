"""Main FISTA orchestrator. Port of reconstruct.m."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import scipy.io
import torch

from odt.io.checkpoint import save_checkpoint
from odt.io.load_data import load_data
from odt.optim.fista import fista_step
from odt.optim.sampling import sample_angles
from odt.physics.bpm_adjoint import bpm_adjoint
from odt.physics.bpm_forward import bpm_forward
from odt.physics.kernels import build_kernels
from odt.preprocess.na_filter import angle_na_from_illumination, filter_angles_by_na
from odt.preprocess.pipeline import apply_pipeline
from odt.reg.pnp import pnp_denoise
from odt.reg.tv_3d import denoise_tv_3d
from odt.reg.tv_cost import tv_cost
from odt.util.backend import get_device, to_device, to_numpy, zeros
from odt.util.validate_config import validate_config


@dataclass
class ReconstructResult:
    """Output of :func:`reconstruct`."""
    x: np.ndarray                       # final reconstruction (Ny, Nx, Nz), real
    cost:    np.ndarray                 # (n_iter,) total cost
    cost_d:  np.ndarray                 # (n_iter,) data fidelity
    cost_r:  np.ndarray                 # (n_iter,) regularization
    mse:     np.ndarray                 # (n_iter,) zeros unless ground truth wired in
    ang_set: np.ndarray                 # (ang_num, n_iter) sampled angles per iter
    cfg:     dict                       # echo of input config
    diagnostics: Optional[dict] = None  # per-iter stopping criteria (if enabled)
    kept_angle_idx: Optional[np.ndarray] = None  # if partial_NA filter ran
    kept_angle_NA:  Optional[np.ndarray] = None


def reconstruct(cfg: Mapping) -> ReconstructResult:
    """Run 3D ODT reconstruction (FISTA + 3D TV + BPM).

    See ``configs/default.py`` for the schema.
    """
    validate_config(cfg)

    Nx = int(cfg["geom"]["Nx"])
    Ny = int(cfg["geom"]["Ny"])
    Nz = int(cfg["geom"]["Nz"])

    n_iter      = int(cfg["optim"]["n_iter"])
    n_iter_tv   = int(cfg["reg"]["n_iter_tv"])
    gamma       = float(cfg["optim"]["gamma"])
    damping     = float(cfg["optim"]["damping"])
    ang_num     = int(cfg["optim"]["ang_num"])
    stochastic  = bool(cfg["optim"]["stochastic"])
    tau         = float(cfg["reg"]["tau"])
    n_upper     = float(cfg["reg"]["n_upper"])
    n_lower     = float(cfg["reg"].get("n_lower", 0.0))
    clamp_neg   = bool(cfg["reg"].get("clamp_negative_after_tv", True))
    prox_kind   = str(cfg["reg"].get("prox_kind", "tv")).lower()
    pnp_sigma   = float(cfg["reg"].get("pnp_sigma", 0.0))

    viz_cfg     = cfg.get("viz", {}) or {}
    viz_enable  = bool(viz_cfg.get("enable", False))   # default OFF in Python (headless friendly)
    save_every  = bool(viz_cfg.get("save_every", False))
    checkpoints = list(cfg["io"].get(
        "checkpoint_at",
        [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 700, 1000],
    ))
    log_diagnostics = bool(cfg.get("log_diagnostics", False))

    output_dir = cfg["io"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print(f"[odt] device: {device}, n_iter={n_iter}, ang_num={ang_num}, "
          f"stochastic={stochastic}, tau={tau}")

    # --- Load + preprocess + move to GPU ---------------------------------
    UI_stack_np, U_stack_np = load_data(
        cfg["io"]["data_path"],
        cfg["io"].get("data_variant", "auto"),
    )

    # Optional NA filter: keep only illumination angles within [min_NA, max_NA].
    pp_na = (cfg.get("preprocess", {}) or {}).get("partial_NA", {}) or {}
    kept_idx_np = None
    kept_NA_np  = None
    if pp_na.get("enable", False):
        NA_per_angle = angle_na_from_illumination(
            UI_stack_np,
            wl=float(cfg["physics"]["wl"]),
            npix=float(cfg["physics"]["npix"]),
        )
        max_NA_cfg = float(pp_na["max_NA"])
        min_NA_cfg = float(pp_na.get("min_NA", 0.0))
        UI_stack_np, U_stack_np, kept_idx_np, kept_NA_np = filter_angles_by_na(
            UI_stack_np, U_stack_np, NA_per_angle,
            max_NA=max_NA_cfg, min_NA=min_NA_cfg,
        )
        print(
            f"[odt] partial_NA: kept {len(kept_idx_np)}/{len(NA_per_angle)} angles "
            f"in NA ∈ [{min_NA_cfg:.3f}, {max_NA_cfg:.3f}] "
            f"(actual range {kept_NA_np.min():.3f}-{kept_NA_np.max():.3f})"
        )

    # Crop to (Ny, Nx) if larger
    Ny_in, Nx_in, _ = U_stack_np.shape
    if Ny_in != Ny or Nx_in != Nx:
        cy = (Ny_in - Ny) // 2
        cx = (Nx_in - Nx) // 2
        U_stack_np  = U_stack_np [cy: cy + Ny, cx: cx + Nx, :]
        UI_stack_np = UI_stack_np[cy: cy + Ny, cx: cx + Nx, :]

    UI_stack = to_device(UI_stack_np, dtype=torch.complex64)
    U_stack  = to_device(U_stack_np,  dtype=torch.complex64)

    # Preprocess pipeline (window, normalize, phase-ramp, unwrap).
    # If global_max normalization is active, capture the scalar M=max|U| and
    # rescale (gamma, tau) so the FISTA optimizer behaves identically to the
    # no-norm case:
    #   * L_d scales by 1/M^2 (because both U and UI are divided by M and
    #     BPM is linear in field amplitude). To recover the same gradient
    #     step  x -= gamma * grad,  gamma must be MULTIPLIED by M^2.
    #   * TV(x) is unchanged. To keep the data-vs-TV weighting constant in
    #     the total cost L = L_d + tau*TV, tau must be DIVIDED by M^2.
    #   * The prox-TV strength inside FISTA is gamma*tau, which is invariant
    #     under these rescalings (gamma*M^2 * tau/M^2 = gamma*tau). Good.
    _pp_cfg = cfg.get("preprocess", {})
    _norm_mode = _pp_cfg.get("intensity_normalize", False)
    _M_scale = float(U_stack.abs().max().item()) if _norm_mode == "global_max" else 1.0
    U_stack, UI_stack = apply_pipeline(U_stack, UI_stack, _pp_cfg)
    if _M_scale != 1.0:
        _M2 = _M_scale ** 2
        gamma *= _M2
        tau   /= _M2
        print(f"[odt] global_max norm active: M=max|U|={_M_scale:.4f}, M^2={_M2:.3f}; "
              f"gamma *= M^2 (new gamma={gamma:.3e}), "
              f"tau /= M^2 (new tau={tau:.3e})")

    # MATLAB:  ifftshift(ifftshift(*, 1), 2)  → torch dim=(0, 1)
    UI_stack = torch.fft.ifftshift(UI_stack, dim=(0, 1))
    U_stack  = torch.fft.ifftshift(U_stack,  dim=(0, 1))

    total_ang = UI_stack.shape[2]
    if ang_num > total_ang:
        raise ValueError(
            f"optim.ang_num ({ang_num}) > available angles ({total_ang})."
        )

    # --- Kernels ---------------------------------------------------------
    kernels = build_kernels(cfg["physics"], cfg["geom"])
    DFR     = kernels.DFR
    DFR_Bpr = kernels.DFR_Bpr
    DFR_Mea = kernels.DFR_Mea
    DFR_Ori = kernels.DFR_Ori
    k0      = kernels.k0
    dz      = kernels.dz

    # --- State -----------------------------------------------------------
    x_curr     = zeros((Ny, Nx, Nz), dtype=torch.float32)
    x_prev     = zeros((Ny, Nx, Nz), dtype=torch.float32)
    x_momentum = zeros((Ny, Nx, Nz), dtype=torch.float32)
    q_old      = 1.0

    # Optional inline Rytov initialisation: runs RytovSolver on the
    # just-loaded (UI, U) fields, writes the result as an .npy next to
    # the recon output, and routes it through the existing init_path
    # mechanism below.
    rytov_cfg = (cfg["optim"].get("rytov_init") or {})
    if rytov_cfg.get("enable", False):
        from odt.init_rytov import rytov_init
        print("[odt] running Rytov init...")
        RI_real = rytov_init(UI_stack_np, U_stack_np, cfg)
        save_path = rytov_cfg.get("save_path") or os.path.join(
            output_dir, "rytov_init.npy"
        )
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.save(save_path, RI_real)
        print(f"[odt] Rytov init saved -> {save_path}  "
              f"range [{RI_real.min():+.4f}, {RI_real.max():+.4f}]")
        cfg["optim"]["init_path"] = save_path
        # The Rytov output is absolute RI in natural-image coords.
        # Use the existing init machinery defaults: subtract n0 to get
        # Δn, and ifftshift lateral to DC-at-corner (BPM convention).
        cfg["optim"].setdefault("init_subtract_n0", True)
        cfg["optim"].setdefault("init_fftshift_lateral", True)

    # Optional initialisation from external .npy (e.g. Rytov estimate)
    init_path = cfg["optim"].get("init_path")
    if init_path:
        init_arr = np.load(str(init_path))
        if cfg["optim"].get("init_subtract_n0", True):
            init_arr = init_arr - float(cfg["physics"]["n0"])
        if cfg["optim"].get("init_fftshift_lateral", True):
            init_arr = np.fft.ifftshift(np.fft.ifftshift(init_arr, axes=0), axes=1)
        tr = cfg["optim"].get("init_transpose")
        if tr:
            init_arr = np.transpose(init_arr, tr)
        if init_arr.shape != (Ny, Nx, Nz):
            raise ValueError(
                f"init array shape {init_arr.shape} != reconstruction grid "
                f"({Ny}, {Nx}, {Nz}); set optim.init_transpose to fix axis order"
            )
        init_t = to_device(init_arr.astype(np.float32), dtype=torch.float32)
        x_curr.copy_(init_t)
        x_prev.copy_(init_t)
        x_momentum.copy_(init_t)   # iter-1 BPM forward uses x_momentum
        print(f"[odt] initialised x from {init_path}  range "
              f"[{float(init_t.min()):+.4f}, {float(init_t.max()):+.4f}]")

    cost    = np.zeros(n_iter, dtype=np.float64)
    cost_d  = np.zeros(n_iter, dtype=np.float64)
    cost_r  = np.zeros(n_iter, dtype=np.float64)
    mse     = np.zeros(n_iter, dtype=np.float64)
    ang_set = np.zeros((ang_num, n_iter), dtype=np.int64)

    # Preallocated GPU buffers (reused across iterations)
    PHM        = zeros((Ny, Nx, Nz), dtype=torch.complex64)
    y_field    = zeros((Ny, Nx, Nz), dtype=torch.complex64)
    s          = zeros((Ny, Nx, Nz), dtype=torch.complex64)
    grad_accum = zeros((Ny, Nx, Nz), dtype=torch.complex64)

    # Diagnostics arrays (stopping criteria logged per iteration)
    diag = None
    if log_diagnostics:
        from odt.optim.stopping import (
            relative_reconstruction_change,
            morozov_discrepancy,
            estimate_noise_sigma,
            smoothed_cost_slope,
            residual_entropy as _residual_entropy_fn,
            reconstruction_entropy as _recon_entropy_fn,
            snr_peak,
            tv_relative_change,
        )
        diag = {
            "rel_change":    np.full(n_iter, np.nan),
            "morozov":       np.full(n_iter, np.nan),
            "cost_smooth_slope": np.full(n_iter, np.nan),
            "residual_H":    np.full(n_iter, np.nan),
            "recon_H":       np.full(n_iter, np.nan),
            "snr":           np.full(n_iter, np.nan),
            "tv_change":     np.full(n_iter, np.nan),
        }
        # Estimate noise sigma from the corner regions of U_stack
        _noise_sigma = estimate_noise_sigma(U_stack)
        print(f"[diag] estimated noise sigma = {_noise_sigma:.6g}")

    # Optional viz
    fig = None
    if viz_enable:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(16, 10))

    # Save the resolved config alongside outputs
    scipy.io.savemat(
        os.path.join(output_dir, "config.mat"),
        {"cfg": _flatten_cfg(cfg)},
    )

    # --- FISTA loop ------------------------------------------------------
    rng = np.random.default_rng() if stochastic else None
    for it in range(n_iter):
        print(f"{it + 1}-th iteration is going on")
        gamma = gamma * damping
        grad_accum.zero_()

        sub_ang = sample_angles(total_ang, ang_num, it, stochastic, rng=rng)
        ang_set[:, it] = sub_ang + 1   # store as 1-based to match MATLAB output

        t0 = time.perf_counter()
        for a in range(ang_num):
            ang = int(sub_ang[a])
            cos_factor = 1.0     # single-axis illumination assumption (matches original)

            # --- Forward (BPM) --------------------------------------
            u_inc_ori = torch.fft.ifft2(
                torch.fft.fft2(UI_stack[:, :, ang], dim=(0, 1)) * DFR_Ori,
                dim=(0, 1),
            )
            PHM = torch.exp(1j * (k0 * dz / cos_factor) * x_momentum.to(torch.complex64))
            u_out, y_field = bpm_forward(u_inc_ori, PHM, DFR, y_field)

            # --- Adjoint (gradient + per-angle data cost) ----------
            u_meas_back = torch.fft.ifft2(
                torch.fft.fft2(U_stack[:, :, ang], dim=(0, 1)) * DFR_Mea,
                dim=(0, 1),
            )
            grad_contrib, r_norm_sq = bpm_adjoint(
                u_out, u_meas_back, y_field, PHM, DFR, DFR_Bpr,
                k0, dz, cos_factor, s,
            )
            grad_accum = grad_accum + grad_contrib
            cost_d[it] += r_norm_sq
        torch.cuda.synchronize() if device.type == "cuda" else None
        print(f"  Elapsed: {time.perf_counter() - t0:.3f} sec")

        grad_accum_real = grad_accum.real

        # --- Costs --------------------------------------------------
        cost_d[it] = float(0.5 / ang_num * cost_d[it])
        cost_r[it] = tau * tv_cost(x_prev)
        cost[it]   = cost_d[it] + cost_r[it]

        # --- Proximal step (gradient + regularizer) ----------------
        z_step = x_momentum - (gamma / ang_num) * grad_accum_real
        if prox_kind != "tv":
            # Plug-and-Play denoiser (BM3D/BM4D/NLM) replaces the TV prox.
            x_curr = pnp_denoise(
                z_step, pnp_sigma, kind=prox_kind,
                box=(n_lower, n_upper),
            )
            if clamp_neg:
                x_curr = torch.clamp(x_curr, min=0.0)
        elif tau == 0:
            x_curr = z_step
        else:
            x_curr, _, _, _ = denoise_tv_3d(
                z_step, gamma * tau,
                maxiter=n_iter_tv,
                bounds=(n_lower, n_upper),
                verbose=False,
            )
            if clamp_neg:
                x_curr = torch.clamp(x_curr, min=0.0)

        # --- FISTA momentum update --------------------------------
        x_momentum, q_old = fista_step(x_curr, x_prev, q_old)

        # --- Diagnostics (stopping criteria) ----------------------
        if diag is not None:
            diag["rel_change"][it] = relative_reconstruction_change(x_curr, x_prev)
            diag["morozov"][it] = morozov_discrepancy(
                cost_d[it], ang_num * Ny * Nx, _noise_sigma,
            )
            diag["cost_smooth_slope"][it] = smoothed_cost_slope(cost, it, window=10)
            diag["recon_H"][it] = _recon_entropy_fn(x_curr)
            diag["snr"][it] = snr_peak(x_curr)
            diag["tv_change"][it] = tv_relative_change(x_curr, x_prev)
            # Residual entropy: use the last angle's residual as a proxy
            # (computing all angles' residuals each iter is too expensive)
            last_ang = int(sub_ang[-1])
            u_inc_diag = torch.fft.ifft2(
                torch.fft.fft2(UI_stack[:, :, last_ang], dim=(0, 1)) * DFR_Ori,
                dim=(0, 1),
            )
            PHM_diag = torch.exp(1j * (k0 * dz) * x_curr.to(torch.complex64))
            u_out_diag = u_inc_diag
            for zi in range(Nz):
                u_out_diag = torch.fft.ifft2(
                    torch.fft.fft2(u_out_diag, dim=(0, 1)) * DFR, dim=(0, 1)
                ) * PHM_diag[:, :, zi]
            u_meas_diag = torch.fft.ifft2(
                torch.fft.fft2(U_stack[:, :, last_ang], dim=(0, 1)) * DFR_Mea,
                dim=(0, 1),
            )
            r_diag = u_out_diag - u_meas_diag
            diag["residual_H"][it] = _residual_entropy_fn(r_diag)

        x_prev = x_curr

        # --- Visualization + checkpoints --------------------------
        if viz_enable and fig is not None:
            from odt.viz.convergence import plot_convergence
            from odt.viz.slices import show_slices
            fig.clf()
            plot_convergence(it + 1, mse, cost, fig=fig)
            show_slices(x_curr, it + 1, fig=fig, viz_cfg=viz_cfg)
            if save_every:
                fig.savefig(os.path.join(output_dir, f"Epoch_{it + 1}.png"), dpi=80)

        if (it + 1) in checkpoints:
            save_checkpoint(
                output_dir, it + 1, x_curr,
                cost=cost, cost_d=cost_d, cost_r=cost_r,
                ang_set=ang_set, cfg=cfg,
            )
            if viz_cfg.get("save_checkpoint_plot", True):
                from odt.viz.checkpoint_plot import save_checkpoint_plot
                save_checkpoint_plot(
                    x_curr, it + 1, output_dir,
                    cmap=viz_cfg.get("cmap", "gray"),
                    caxis_range=viz_cfg.get("caxis_range"),
                    aspect=viz_cfg.get("aspect", "equal"),
                    npix=cfg["physics"].get("npix"),
                    dz=cfg["physics"].get("dz"),
                    scalebar_um=viz_cfg.get("scalebar_um"),
                    show_crosshairs=viz_cfg.get("show_crosshairs", True),
                    show_cube_inset=viz_cfg.get("show_cube_inset", False),
                    center_focus=bool(viz_cfg.get("center_focus", False)),
                    iz_focus=cfg.get("geom", {}).get("iz_focus"),
                    fftshift_z=bool(viz_cfg.get("fftshift_z", False)),
                )

    # --- Optional z-scan video at end ----------------------------------
    if viz_cfg.get("save_video_at_end", False):
        from odt.viz.video import save_z_scan_video
        video_path = os.path.join(output_dir, "z_scan.mp4")
        try:
            save_z_scan_video(
                x_curr, video_path,
                cmap=viz_cfg.get("cmap", "gray"),
                caxis_range=viz_cfg.get("caxis_range"),
                fps=int(viz_cfg.get("video_fps", 12)),
                title_prefix=f"final  ",
            )
            print(f"[odt] z-scan video saved: {video_path}")
        except Exception as exc:
            # Fall back to GIF if MP4 writer fails
            print(f"[odt] MP4 write failed ({exc}); falling back to GIF")
            save_z_scan_video(
                x_curr, video_path.replace(".mp4", ".gif"),
                cmap=viz_cfg.get("cmap", "gray"),
                caxis_range=viz_cfg.get("caxis_range"),
                fps=int(viz_cfg.get("video_fps", 12)),
                title_prefix=f"final  ",
            )

    # --- Optional sweep video (xy changes per frame) -------------------
    if viz_cfg.get("save_sweep_video", False):
        from odt.viz.sweep_video import save_sweep_video
        sweep_path = os.path.join(output_dir, "sweep.mp4")
        try:
            save_sweep_video(
                x_curr, sweep_path,
                cmap=viz_cfg.get("cmap", "gray"),
                caxis_range=viz_cfg.get("caxis_range"),
                fps=int(viz_cfg.get("video_fps", 12)),
                title_prefix="final  ",
                aspect=viz_cfg.get("aspect", "equal"),
                npix=cfg["physics"].get("npix"),
                scalebar_um=viz_cfg.get("scalebar_um"),
            )
            print(f"[odt] sweep video saved: {sweep_path}")
        except Exception as exc:
            print(f"[odt] sweep MP4 write failed ({exc}); falling back to GIF")
            save_sweep_video(
                x_curr, sweep_path.replace(".mp4", ".gif"),
                cmap=viz_cfg.get("cmap", "gray"),
                caxis_range=viz_cfg.get("caxis_range"),
                fps=int(viz_cfg.get("video_fps", 12)),
                title_prefix="final  ",
                aspect=viz_cfg.get("aspect", "equal"),
                npix=cfg["physics"].get("npix"),
                scalebar_um=viz_cfg.get("scalebar_um"),
            )

    # --- Final result ----------------------------------------------------
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Auto z-scan video at the last checkpoint epoch (default ON; disable
    # via viz.save_zscan_video=false).
    if (cfg.get("viz", {}) or {}).get("save_zscan_video", True):
        try:
            from odt.viz.video import save_z_scan_video
            n_final = max(checkpoints) if checkpoints else n_iter
            out_mp4 = os.path.join(output_dir, f"z_scan_epoch{n_final}.mp4")
            cmap = (cfg.get("viz", {}) or {}).get("cmap", "gray")
            fps = int((cfg.get("viz", {}) or {}).get("video_fps", 12))
            print(f"[viz] rendering z-scan video -> {out_mp4}  (cmap={cmap}, fps={fps})")
            save_z_scan_video(to_numpy(x_curr), out_mp4,
                              cmap=cmap, fps=fps, apply_fftshift=True)
        except Exception as e:
            print(f"[viz] z-scan video failed: {e}")

    return ReconstructResult(
        x=to_numpy(x_curr),
        cost=cost,
        cost_d=cost_d,
        cost_r=cost_r,
        mse=mse,
        ang_set=ang_set,
        cfg=_flatten_cfg(cfg),
        diagnostics=diag,
        kept_angle_idx=kept_idx_np,
        kept_angle_NA=kept_NA_np,
    )


def _flatten_cfg(cfg: Mapping) -> dict:
    """Convert nested config dict into a savemat-friendly dict (no None)."""
    out = {}
    for k, v in cfg.items():
        if isinstance(v, Mapping):
            out[k] = _flatten_cfg(v)
        elif v is None:
            continue
        else:
            out[k] = v
    return out
