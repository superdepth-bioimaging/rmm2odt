"""End-to-end orchestrator: rrmat -> multilayer -> GVAS -> BPM_ODT -> image.

Chains the four vendored stages, passing file artifacts between them. Heavy
stage modules (torch) are imported lazily inside each runner so ``--dry-run``
needs no GPU / torch.

Artifact flow (all under paths.output_root):
  rrmat.mat
    -> multilayer_base/ (layers+pupils)            [skipped if multilayer.init_dir set]
    -> multilayer_scan/output_layer_stack.npy + depths.npy
    -> gvas/odt_input.mat
    -> odt/ss_opt_epochN.mat (+ final.mat)
    -> render/fig_ortho.{png,svg} + ri_zscan.mp4
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import rmm_config as C


def _log(msg):
    print(f"[rmm2odt] {msg}", flush=True)


def _check(path, label):
    ok = os.path.exists(path)
    _log(f"  {'OK ' if ok else 'MISSING'} {label}: {path}")
    return ok


def _stage_collapsed_init(base_dir, collapse):
    """Build a reduced-model init dir from the full base recon.

    Frozen (outside) layers are copied from the base by position; the single
    object layer gets a placeholder file (the solver resets the scanning layer
    to flat on load). Reflection geometry: N output layers, N-1 input layers
    (the last layer = sample plane has no input layer)."""
    import glob
    import shutil
    base = Path(base_dir)
    stage = base.parent / "scan_init_collapsed"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    E = collapse["base_epoch"]
    reduced = collapse["reduced"]
    obj_idx = collapse["object_idx"]
    fmap = dict(collapse["frozen_map"])          # reduced_idx -> base output idx
    N = len(reduced)
    ph_out = base / f"output_layer_1_epoch_{E}.txt"   # placeholder (reset to flat)
    ph_in = base / f"input_layer_1_epoch_{E}.txt"
    for r in range(N):
        src = ph_out if r == obj_idx else base / f"output_layer_{fmap[r] + 1}_epoch_{E}.txt"
        shutil.copyfile(src, stage / f"output_layer_{r + 1}_epoch_{E}.txt")
        if r < N - 1:                            # no input layer at the sample plane (last)
            srci = ph_in if r == obj_idx else base / f"input_layer_{fmap[r] + 1}_epoch_{E}.txt"
            shutil.copyfile(srci, stage / f"input_layer_{r + 1}_epoch_{E}.txt")
    for f in glob.glob(str(Path(glob.escape(str(base))) / f"pupil_aberration_*_epoch_{E}.txt")):
        shutil.copyfile(f, stage / Path(f).name)
    return str(stage)


# --------------------------------------------------------------------------
# Stage runners
# --------------------------------------------------------------------------

def run_multilayer(P, d, dry):
    user_init = (P.get("multilayer", {}) or {}).get("init_dir", "") or ""
    if user_init:
        _log(f"stage multilayer: reusing init_dir (skipping base recon): {user_init}")
        init_dir = user_init
    else:
        base = C.build_multilayer_base_cfg(P, d)
        _log(f"stage multilayer: base reconstruction -> {base['output_dir']}")
        if dry:
            _check(base["rrmat_path"], "rrmat")
            _log(f"  base: epochs={base['optimization_epochs']} mode={base['illumination_mode']} "
                 f"layers={base['layer_positions']}")
        else:
            os.makedirs(base["output_dir"], exist_ok=True)
            from multilayer_torch.solver import run_from_config
            run_from_config(base)
        init_dir = d["multilayer_base"]

    scan = C.build_multilayer_scan_cfg(P, d, init_dir)
    collapse = scan.pop("_collapse", None)
    obj_idx = scan["scanning_layer_order"][0]
    obj_z = scan["layer_positions"][obj_idx]      # index is into the (reduced) scan model
    win = P["multilayer"]["scan_range"]
    _log(f"stage multilayer: depth scan -> {scan['output_dir']}")
    if collapse:
        _log(f"  COLLAPSE: {collapse['n_collapsed']} layers inside window {win} -> ONE object "
             f"layer at z={obj_z} (mode={collapse['mode']}); reduced scan model={collapse['reduced']}, "
             f"object idx={obj_idx}; all other layers frozen (carried from base recon).")
    else:
        _log(f"  AUTO object layer: idx={obj_idx} (z={obj_z}) is the unique layer in window {win}; "
             f"all others frozen.")
    _log(f"  dz sweep={scan['scan_range']} step={scan['scan_step']}, epochs={scan['scan_optimization_epochs']}")
    if not dry:
        if collapse:
            scan["init_dir"] = _stage_collapsed_init(init_dir, collapse)
            _log(f"  staged reduced init-dir: {scan['init_dir']}")
        os.makedirs(scan["output_dir"], exist_ok=True)
        from multilayer_torch.solver import run_from_config
        run_from_config(scan)
    out = os.path.join(d["multilayer_scan"], "output_layer_stack.npy")
    _log(f"stage multilayer: depth-scan stack -> {out}")
    return out


def run_gvas(P, d, dry):
    cass = os.path.join(d["multilayer_scan"], "output_layer_stack.npy")
    illum = P["paths"]["illumination_file"]
    out = os.path.join(d["gvas"], "odt_input.mat")
    _log(f"stage gvas: {cass} + illumination -> {out}")
    if dry:
        _check(cass, "scan stack (from stage 1)")
        _check(illum, "illumination_file")
        return out
    cfg = C.build_gvas_config(P, d)
    from gvas_torch.solver import run_optimization
    run_optimization(cfg)
    _log(f"stage gvas: wrote {out}")
    return out


def run_odt(P, d, dry):
    data_path = os.path.join(d["gvas"], "odt_input.mat")
    _log(f"stage odt: {data_path} -> {d['odt']} (ss_opt_epochN.mat + final.mat)")
    cfg = C.build_odt_cfg(P, d)
    if dry:
        _check(data_path, "odt_input.mat (from stage 2)")
        _log(f"  odt: geom={cfg['geom']} n_iter={cfg['optim']['n_iter']} "
             f"tau={cfg['reg']['tau']} checkpoints={cfg['io']['checkpoint_at']}")
        return C.odt_final_checkpoint(P, d)
    os.makedirs(d["odt"], exist_ok=True)
    from odt.io.config_loader import resolve_config
    from odt.reconstruct import reconstruct
    import scipy.io
    cfg = resolve_config(cfg)
    result = reconstruct(cfg)
    final_path = os.path.join(d["odt"], "final.mat")
    scipy.io.savemat(final_path, {"x": result.x, "cost": result.cost,
                                  "cost_d": result.cost_d, "cost_r": result.cost_r,
                                  "ang_set": result.ang_set}, do_compression=True)
    _log(f"stage odt: wrote {final_path}")
    return C.odt_final_checkpoint(P, d)


def run_render(P, d, dry):
    ckpt = C.odt_final_checkpoint(P, d)
    final = os.path.join(d["odt"], "final.mat")
    src = ckpt if os.path.exists(ckpt) else final
    kw = C.build_render_kwargs(P, d)
    do_panels = C.panels_enabled(P)
    _log(f"stage render: {src} -> {kw['out_dir']} (fig_ortho + ri_zscan"
         f"{' + manuscript panels' if do_panels else ''})")
    if dry:
        if not (_check(ckpt, "ss_opt checkpoint") or _check(final, "final.mat")):
            _log("  (render input absent until stage 3 runs)")
        if do_panels:
            pk = C.build_panels_kwargs(P, d)
            _log("  manuscript panels -> fig2c_confocal_reflectance, "
                 "fig2e_optimized_layers, fig2e_optimized_object, "
                 "fig3b_depth_scanned_layers, fig3d_angular_transmission_fields, "
                 f"fig4f_multi_depth_xy_odt_images (fmt={pk['formats']})")
            _check(pk["rrmat"], "fig2c_confocal_reflectance: rrmat")
            _check(pk["layer_dir"], "fig2e_optimized_layers/object: base layer dir")
            _check(pk["scan_stack"], "fig3b_depth_scanned_layers: depth-scan stack")
            _check(pk["odt_input"], "fig3d_angular_transmission_fields: odt_input.mat")
            _check(pk["ss_opt"], "fig4f_multi_depth_xy_odt_images: ss_opt checkpoint")
            _log(f"  fig4f (grayscale) z_um={pk['fig4_z_um']} cmap={pk['fig4_cmap']}")
        return kw["out_dir"]
    # final.mat stores volume under key 'x'; ss_opt checkpoints under 'ss_opt'
    key = "x" if src.endswith("final.mat") else "ss_opt"
    from rmm_render import render_odt
    written = render_odt(src, ss_opt_key=key, **kw)
    _log(f"stage render: wrote {written}")
    if do_panels:
        from rmm_panels import render_manuscript_panels
        pk = C.build_panels_kwargs(P, d)
        panels = render_manuscript_panels(**pk)
        _log(f"stage render: manuscript panels wrote {sorted(panels)}")
    return kw["out_dir"]


RUNNERS = {"multilayer": run_multilayer, "gvas": run_gvas,
           "odt": run_odt, "render": run_render}


def run_pipeline(cfg_path, from_stage=None, to_stage=None, dry=False):
    P = C.load_pipeline_config(cfg_path)
    d = C.stage_dirs(P)
    if not dry:
        os.makedirs(d["root"], exist_ok=True)
    order = C.STAGES
    i0 = order.index(from_stage) if from_stage else 0
    i1 = order.index(to_stage) if to_stage else len(order) - 1
    if i0 > i1:
        raise ValueError(f"--from {from_stage} is after --to {to_stage}")
    selected = order[i0:i1 + 1]
    _log(f"pipeline '{P.get('name','rmm2odt')}'  stages: {selected}  "
         f"{'(DRY RUN)' if dry else ''}")
    _log(f"output_root: {d['root']}")
    for st in selected:
        RUNNERS[st](P, d, dry)
    _log("done." + ("  (dry run - nothing executed)" if dry else ""))
