"""Unified pipeline config: load one YAML, build each stage's native config.

One ``pipeline.yaml`` (see configs/example_pipeline.yaml) drives all four
stages. This module loads it, resolves per-stage output directories under a
single ``paths.output_root``, and builds the exact config object each stage's
entry point expects:

  stage 1  multilayer.solver.run_from_config(dict)   -- base recon + depth scan
  stage 2  gvas.solver.run_optimization(Config)      -- typed dataclass
  stage 3  odt.reconstruct.reconstruct(dict)         -- via resolve_config
  stage 4  rmm_render.render_odt(...)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

STAGES = ["multilayer", "gvas", "odt", "render"]

_ENV_RE = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _expand(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), obj)
    return obj


def load_pipeline_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = _expand(raw)
    for sec in ("optics", "paths"):
        if sec not in raw:
            raise ValueError(f"pipeline config missing required section '{sec}'")
    if "rrmat" not in raw["paths"] or "output_root" not in raw["paths"]:
        raise ValueError("paths must define 'rrmat' and 'output_root'")
    return raw


def stage_dirs(P: dict) -> dict:
    root = P["paths"]["output_root"]
    return {
        "root": root,
        "multilayer_base": os.path.join(root, "multilayer_base"),
        "multilayer_scan": os.path.join(root, "multilayer_scan"),
        "gvas_root": root,            # GVAS writes to <output_dir>/<run_name>
        "gvas": os.path.join(root, "gvas"),
        "odt": os.path.join(root, "odt"),
        "render": os.path.join(root, "render"),
    }


def _optics(P):
    o = P["optics"]
    return dict(wavelength=float(o["wavelength"]), NA=float(o["NA"]),
                n0=float(o["n0"]), resolution=float(o["resolution"]),
                scan_step=float(o.get("scan_step", 1.0)),
                NA_model=float(o.get("NA_model", o["NA"])))


# --------------------------------------------------------------------------
# Stage 1: multilayer (base recon + depth scan) -> run_from_config(dict)
# --------------------------------------------------------------------------

def _multilayer_common(P, d):
    o = _optics(P)
    m = P.get("multilayer", {})
    cfg = dict(
        wavelength=o["wavelength"], NA=o["NA"], NA_model=o["NA_model"],
        medium_refractive_index=o["n0"], resolution=o["resolution"],
        layer_positions=list(m["layer_positions"]),
        pupil_funcs_type=list(m.get("pupil_funcs_type", ["variable", "variable"])),
        optimization_hyperparams=dict(m.get("optimization_hyperparams",
            {"batch_size": 50, "learning_rate": 0.02,
             "gamma_coeff_1": 1.0, "gamma_coeff_2": 0.0})),
        geometry_mode=m.get("geometry_mode", "Reflection"),
        complex_tensor_type=m.get("complex_tensor_type", "Rectangular"),
        reg_mode=m.get("reg_mode", ["L2_norm"]),
        scanning_size=int(m.get("scanning_size", 0)),
        ROD=m.get("ROD", None),
        data_format="rrmat",
        rrmat_path=P["paths"]["rrmat"],
        rrmat_sigma=float(m.get("rrmat_sigma", 0)),
        rrmat_spot_offset=tuple(int(v) for v in m.get("rrmat_spot_offset", (0, 0))),
        save_every=int(m.get("save_every", 5)),
    )
    spk = m.get("speckle")
    if spk:
        cfg["illumination_mode"] = m.get("illumination_mode", "speckle")
        cfg["speckle"] = dict(spk)
    else:
        cfg["illumination_mode"] = m.get("illumination_mode", "single_focus")
    return cfg


def build_multilayer_base_cfg(P, d):
    m = P.get("multilayer", {})
    cfg = _multilayer_common(P, d)
    cfg.update(
        is_scan=False,
        output_dir=d["multilayer_base"],
        optimization_epochs=int(m.get("base_epochs", m.get("optimization_epochs", 20))),
        layers_init="flat",
    )
    return cfg


def derive_scan_layer(layer_positions, scan_window):
    """Auto-pick the scanning ('object') layer from an ABSOLUTE z window.

    The scanning layer is the UNIQUE layer whose position lies inside the
    half-open window [z_lo, z_hi). All other layers are frozen. Returns
    ``(obj_idx, base_z, dz_range)`` where ``dz_range = [z_lo - base_z,
    z_hi - base_z]`` -- the relative shifts that sweep the object across the
    window (fed to the solver, which does ``np.arange(dz_lo, dz_hi, step)``).

    Raises ValueError unless EXACTLY ONE layer is inside the window.
    """
    positions = [float(z) for z in layer_positions]
    z_lo, z_hi = float(scan_window[0]), float(scan_window[1])
    if z_hi <= z_lo:
        raise ValueError(f"scan_range must be [z_lo, z_hi] with z_hi > z_lo; got {scan_window}")
    inside = [i for i, z in enumerate(positions) if z_lo <= z < z_hi]
    if len(inside) != 1:
        found = [(i, positions[i]) for i in inside]
        raise ValueError(
            f"scan window [{z_lo}, {z_hi}) must contain EXACTLY ONE layer position; "
            f"found {len(inside)}: {found}. layer_positions={positions}. "
            f"Adjust layer_positions or scan_range so a single object layer sits inside."
        )
    obj_idx = inside[0]
    base_z = positions[obj_idx]
    return obj_idx, base_z, [z_lo - base_z, z_hi - base_z]


def collapse_scan_model(layer_positions, scan_window, mode="center"):
    """SCAN-ONLY model: replace the (>=2) in-window layers with ONE object layer.

    The base reconstruction keeps the full model; only the depth scan uses this
    reduced model = all outside layers (frozen, carried from the base recon) +
    ONE object layer that is swept across the window. Returns
    ``(reduced_positions, object_idx, dz_range, frozen_map)``:
      reduced_positions : sorted-descending list (sample plane / z=min last)
      object_idx        : index of the object layer in reduced_positions
      dz_range          : [z_lo-obj_z, z_hi-obj_z] (sweep so object covers window)
      frozen_map        : list of (reduced_idx, base_idx) for layers copied from
                          the full base model (object excluded)
    """
    positions = [float(z) for z in layer_positions]
    z_lo, z_hi = float(scan_window[0]), float(scan_window[1])
    inside = [i for i, z in enumerate(positions) if z_lo <= z < z_hi]
    if len(inside) < 2:
        raise ValueError("collapse_scan_model needs >=2 in-window layers")
    if mode == "center":
        obj_z = 0.5 * (z_lo + z_hi)
    elif mode == "bottom":
        obj_z = z_lo
    elif mode == "mean":
        obj_z = sum(positions[i] for i in inside) / len(inside)
    else:
        raise ValueError(f"unknown scan_collapse_mode {mode!r} (center|bottom|mean)")
    outside = [(i, positions[i]) for i in range(len(positions)) if i not in inside]
    if any(abs(p - obj_z) < 1e-9 for _, p in outside):
        raise ValueError(f"object z={obj_z} coincides with an outside layer position; "
                         f"pick a different window or collapse mode")
    reduced = sorted([p for _, p in outside] + [obj_z], reverse=True)
    object_idx = reduced.index(obj_z)
    frozen_map = []
    for r, p in enumerate(reduced):
        if r == object_idx:
            continue
        base_idx = next(i for i, q in outside if abs(q - p) < 1e-9)
        frozen_map.append((r, base_idx))
    return reduced, object_idx, [z_lo - obj_z, z_hi - obj_z], frozen_map


def build_multilayer_scan_cfg(P, d, init_dir):
    """Depth-scan config, AUTO-derived from layer_positions + the absolute scan
    window (multilayer.scan_range = [z_lo, z_hi] um).

    * exactly 1 layer in window -> sweep it, warm-start from the base recon as-is.
    * >=2 layers in window -> collapse to ONE object layer (scan-only reduced
      model); cfg['_collapse'] carries the staging recipe for the orchestrator.
    Always exactly one object layer ends up scanned; everything else frozen.
    """
    m = P.get("multilayer", {})
    base_epochs = int(m.get("base_epochs", m.get("optimization_epochs", 20)))
    if "scan_range" not in m:
        raise ValueError("multilayer.scan_range (absolute z window [z_lo, z_hi], um) is required")
    positions = [float(z) for z in m["layer_positions"]]
    z_lo, z_hi = float(m["scan_range"][0]), float(m["scan_range"][1])
    inside = [i for i, z in enumerate(positions) if z_lo <= z < z_hi]
    if len(inside) == 0:
        raise ValueError(f"scan window [{z_lo}, {z_hi}) contains no layer_position; {positions}")

    cfg = _multilayer_common(P, d)
    collapse = None
    if len(inside) == 1:
        obj_idx, _base_z, dz_range = derive_scan_layer(positions, m["scan_range"])
        cfg["layer_positions"] = positions
    else:
        mode = m.get("scan_collapse_mode", "center")
        reduced, obj_idx, dz_range, frozen_map = collapse_scan_model(positions, m["scan_range"], mode)
        cfg["layer_positions"] = reduced            # SCAN uses the reduced model
        collapse = dict(reduced=reduced, object_idx=obj_idx, frozen_map=frozen_map,
                        full=positions, mode=mode, base_epoch=base_epochs,
                        n_collapsed=len(inside))

    cfg.update(
        is_scan=True,
        output_dir=d["multilayer_scan"],
        init_dir=init_dir,                 # orchestrator overrides with staged dir if collapsing
        layers_init="previous",
        pupil_init="previous",
        pupil_funcs_type=["constant", "constant"],
        init_epoch=base_epochs,
        pupil_filename=m.get("pupil_filename", f"pupil_aberration_*_epoch_{base_epochs}.txt"),
        scanning_layer_order=[obj_idx],
        scan_range=dz_range,
        scan_step=float(m.get("scan_step", 1)),
        scan_optimization_epochs=int(m.get("scan_epochs", 10)),
    )
    cfg["_collapse"] = collapse            # consumed (popped) by the orchestrator
    return cfg


# --------------------------------------------------------------------------
# Stage 2: GVAS -> run_optimization(Config)
# --------------------------------------------------------------------------

def build_gvas_config(P, d):
    """Construct the GVAS typed Config from the unified YAML."""
    from gvas_torch.config import (Config, OpticsConfig, GridConfig,
                                   DataConfig, OptConfig, RuntimeConfig, IOConfig)
    o = _optics(P)
    g = P.get("gvas", {})
    grid = g.get("grid", {"Nx": 300, "Ny": 300, "Nz": 25, "crop_size": 150})
    opt = g.get("optimisation", {})
    data_kw = dict(
        illumination_file=P["paths"]["illumination_file"],
        illumination_mat_key=P["paths"].get("illumination_mat_key"),
        cass_file=os.path.join(d["multilayer_scan"], "output_layer_stack.npy"),
        extend_illumination=bool(g.get("extend_illumination", True)),
    )
    # Depth axis: prefer the depths.npy produced by the multilayer scan.
    if g.get("use_scan_depths", True):
        data_kw["depths_file"] = os.path.join(d["multilayer_scan"], "depths.npy")
        data_kw["depths_center"] = float(g.get("depths_center", 0.0))
    else:
        dr = g.get("distance_range", [-12, 13])
        data_kw["distance_range"] = (int(dr[0]), int(dr[1]))
        data_kw["distance_step"] = int(g.get("distance_step", 1))
    cfg = Config(
        name=P.get("name", "rmm2odt") + "_gvas",
        optics=OpticsConfig(wavelength=o["wavelength"], NA=o["NA"],
                            n0=o["n0"], scan_step=o["scan_step"]),
        grid=GridConfig(Nx=int(grid["Nx"]), Ny=int(grid["Ny"]),
                        Nz=int(grid["Nz"]), crop_size=grid.get("crop_size")),
        data=DataConfig(**data_kw),
        optimisation=OptConfig(
            n_epochs=int(opt.get("n_epochs", 25)),
            batch_size=int(opt.get("batch_size", 3)),
            learning_rate=float(opt.get("learning_rate", 0.05)),
            loss_fn=opt.get("loss_fn", "MSE"),
            reg_mode=opt.get("reg_mode", "L2"),
            reg_weight=float(opt.get("reg_weight", 0.01)),
            save_crop_size=opt.get("save_crop_size"),
        ),
        runtime=RuntimeConfig(device=P.get("runtime", {}).get("device", "cuda")),
        io=IOConfig(output_dir=d["gvas_root"], run_name="gvas"),
    )
    return cfg


# --------------------------------------------------------------------------
# Stage 3: BPM_ODT -> reconstruct(resolve_config(dict))
# --------------------------------------------------------------------------

def build_odt_cfg(P, d):
    o = _optics(P)
    a = P.get("odt", {})
    geom = a.get("geom", {"Nx": 150, "Ny": 150, "Nz": 150})
    optim = a.get("optim", {})
    reg = a.get("reg", {})
    cfg = {
        "physics": {
            "wl": o["wavelength"], "n0": o["n0"], "NA": o["NA"],
            "npix": float(a.get("npix", o["resolution"])),
            "dz": float(a.get("dz", o["resolution"])),
        },
        "geom": {"Nx": int(geom["Nx"]), "Ny": int(geom["Ny"]), "Nz": int(geom["Nz"])},
        "optim": {
            "n_iter": int(optim.get("n_iter", 50)),
            "gamma": float(optim.get("gamma", 1e-4)),
            "damping": float(optim.get("damping", 1.0)),
            "ang_num": int(optim.get("ang_num", 40)),
            "stochastic": bool(optim.get("stochastic", True)),
        },
        "reg": {
            "tau": float(reg.get("tau", 0.02)),
            "n_iter_tv": int(reg.get("n_iter_tv", 5)),
            "n_upper": float(reg.get("n_upper", 1.7)),
            "n_lower": float(reg.get("n_lower", 0.0)),
        },
        "preprocess": a.get("preprocess", {"intensity_normalize": False, "window": "none"}),
        "io": {
            "data_path": os.path.join(d["gvas"], "odt_input.mat"),
            "data_variant": a.get("data_variant", "illumination_output"),
            "output_dir": d["odt"],
            "checkpoint_at": list(a.get("checkpoint_at", [10, 20, 30, 40, 50])),
        },
        # we render separately in stage 4; keep the BPM run headless/quiet
        "viz": {"enable": False, "save_checkpoint_plot": False, "save_zscan_video": False},
    }
    return cfg


def odt_final_checkpoint(P, d):
    """Path of the last ss_opt checkpoint the BPM run will write."""
    a = P.get("odt", {})
    ckpts = list(a.get("checkpoint_at", [10, 20, 30, 40, 50]))
    n = max(ckpts) if ckpts else int(a.get("optim", {}).get("n_iter", 50))
    return os.path.join(d["odt"], f"ss_opt_epoch{n}.mat")


# --------------------------------------------------------------------------
# Stage 4: render
# --------------------------------------------------------------------------

def build_render_kwargs(P, d):
    o = _optics(P)
    r = P.get("render", {})
    a = P.get("odt", {})
    return dict(
        out_dir=d["render"],
        resolution_um=float(r.get("resolution_um", o["resolution"])),
        dz_um=float(a.get("dz", o["resolution"])),
        scalebar_um=float(r.get("scalebar_um", 10.0)),
        cmap=r.get("cmap", "inferno"),
        zscan_video=bool(r.get("zscan_video", True)),
        make_ortho=bool(r.get("fig_ortho", True)),
    )


def panels_inputs(P, d):
    """Resolve the input artifact path for each manuscript panel."""
    m = P.get("multilayer", {})
    base_epoch = int(m.get("base_epochs", m.get("optimization_epochs", 20)))
    return {
        "rrmat":      P["paths"]["rrmat"],
        "layer_dir":  d["multilayer_base"],
        "layer_epoch": base_epoch,
        "scan_stack": os.path.join(d["multilayer_scan"], "output_layer_stack.npy"),
        "odt_input":  os.path.join(d["gvas"], "odt_input.mat"),
        "ss_opt":     odt_final_checkpoint(P, d),
    }


def build_panels_kwargs(P, d):
    """Build kwargs for rmm_panels.render_manuscript_panels from the unified YAML.

    The ``render.panels`` sub-block toggles/tunes them. Defaults reproduce the
    manuscript panels (svg_panels_sample5_251208): fig4 is grayscale at
    z = 15/20/25 μm, both PNG + SVG written."""
    o = _optics(P)
    r = P.get("render", {})
    pan = r.get("panels", {}) or {}
    a = P.get("odt", {})
    inputs = panels_inputs(P, d)
    ai = pan.get("angle_indices")
    # fig4 reads the ss_opt checkpoint (key 'ss_opt'); fall back to final.mat
    # (key 'x') when the checkpoint was not written.
    ss_opt = inputs["ss_opt"]
    ss_opt_key = "ss_opt"
    if not os.path.exists(ss_opt):
        fallback = os.path.join(d["odt"], "final.mat")
        if os.path.exists(fallback):
            ss_opt, ss_opt_key = fallback, "x"
    # fig3d optional propagated-rrmat comparison rows.
    # panels.fig3d_rrmat: true -> use paths.rrmat; a string -> that path; else off.
    fig3d_rrmat = pan.get("fig3d_rrmat")
    if fig3d_rrmat is True:
        fig3d_rrmat = P["paths"]["rrmat"]
    elif fig3d_rrmat in (False, None):
        fig3d_rrmat = None
    # fig3d input override (e.g. an epoch-specific odt_input built from a per-epoch
    # U_stack). Absolute path, or a filename resolved under the gvas output dir.
    odt_input = inputs["odt_input"]
    fig3d_in = pan.get("fig3d_odt_input")
    if fig3d_in:
        odt_input = fig3d_in if os.path.isabs(fig3d_in) else os.path.join(d["gvas"], fig3d_in)
    return dict(
        out_dir=d["render"],
        rrmat=inputs["rrmat"],
        layer_dir=inputs["layer_dir"],
        layer_epoch=inputs["layer_epoch"],
        scan_stack=inputs["scan_stack"],
        odt_input=odt_input,
        ss_opt=ss_opt,
        ss_opt_key=ss_opt_key,
        resolution_um=float(r.get("resolution_um", o["resolution"])),
        scalebar_um=float(r.get("scalebar_um", 10.0)),
        dz_um=float(a.get("dz", o["resolution"])),
        wavelength_um=o["wavelength"],
        medium_n=o["n0"],
        object_size_1b=int(pan.get("object_size_1b", 120)),
        crop_2d=pan.get("crop_2d"),
        angle_indices=([int(x) for x in ai] if ai is not None else None),
        layer_positions=P.get("multilayer", {}).get("layer_positions"),
        fig3d_rrmat=fig3d_rrmat,
        fig3d_rrmat_propagate_d=pan.get("fig3d_rrmat_propagate_d"),
        fig3d_rrmat_NA=pan.get("fig3d_rrmat_NA"),
        fig3d_show_output_amp=bool(pan.get("fig3d_show_output_amp", True)),
        fig3d_subtract_output_ramp=bool(pan.get("fig3d_subtract_output_ramp", False)),
        fig4_z_um=[float(z) for z in pan.get("fig4_z_um", [15.0, 20.0, 25.0])],
        fig4_cmap=pan.get("fig4_cmap", "gray"),
        fig4_true_depth_focus_um=pan.get("fig4_true_depth_focus_um"),
        fig4_true_depth_ref_idx=pan.get("fig4_true_depth_ref_idx"),
        fig4_vmin=pan.get("fig4_vmin"),
        fig4_vmax=pan.get("fig4_vmax"),
        fig4_outline_color=pan.get("fig4_outline_color"),
        formats=list(pan.get("formats", ["png", "svg"])),
    )


def panels_enabled(P):
    r = P.get("render", {})
    pan = r.get("panels", {})
    if isinstance(pan, dict):
        return bool(pan.get("enable", True))
    return bool(pan)
