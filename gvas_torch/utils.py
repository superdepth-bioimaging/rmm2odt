"""
utils.py
========
Small cross-cutting helpers.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from .config import Config


def default_run_name(cfg: Config) -> str:
    """Compose a run directory name: ``YYYY-MM-DD_configname``.

    If that already exists in the output dir, append a short timestamp
    to avoid collision: ``YYYY-MM-DD_configname_T1430``.
    Hyperparams are in config_resolved.json, not the folder name.
    """
    date = _dt.datetime.now().strftime("%y%m%d")
    name = f"{date}_{cfg.name}"
    out_root = Path(cfg.io.output_dir).expanduser()
    if (out_root / name).exists():
        ts = _dt.datetime.now().strftime("T%H%M")
        name = f"{name}_{ts}"
    return name


def append_results_index(
    results_dir: str | Path,
    run_name: str,
    cfg: Config,
    final_loss: float,
    elapsed: float,
) -> None:
    """Append one row to results_index.csv in the results root dir.

    Creates the file with header if it doesn't exist.
    """
    import csv

    results_dir = Path(results_dir)
    index_path = results_dir / "results_index.csv"
    exists = index_path.exists()

    row = {
        "run_name": run_name,
        "config_name": cfg.name,
        "data_source": cfg.data.cass_file,
        "epochs": cfg.optimisation.n_epochs,
        "loss_fn": cfg.optimisation.loss_fn,
        "final_loss": f"{final_loss:.6e}",
        "elapsed_s": f"{elapsed:.1f}",
        "date": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    with index_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    # Also maintain a markdown index
    md_path = results_dir / "results_index.md"
    md_exists = md_path.exists()
    with md_path.open("a", encoding="utf-8") as f:
        if not md_exists:
            f.write("# GVAS Results Index\n\n")
            f.write("| Run | Config | Data | Epochs | Loss fn | Final loss | Time | Date |\n")
            f.write("|-----|--------|------|--------|---------|------------|------|------|\n")
        f.write(
            f"| {run_name} | {cfg.name} | `{Path(cfg.data.cass_file).parent.name}` "
            f"| {cfg.optimisation.n_epochs} | {cfg.optimisation.loss_fn} "
            f"| {final_loss:.4e} | {elapsed:.0f}s | {_dt.datetime.now().strftime('%Y-%m-%d')} |\n"
        )


def save_odt_input(
    UI_stack: np.ndarray,
    U_stack: np.ndarray,
    output_dir: str | Path,
    filename: str = "odt_input.mat",
) -> Path:
    """Save illumination and reconstructed field as a MATLAB .mat file.

    The output is formatted as input for the next ODT reconstruction
    algorithm, with variables:

    - ``Illumination_stack`` : ``(A, H, W)`` complex64 — illumination fields
    - ``Output_stack``       : ``(A, H, W)`` complex64 — reconstructed U_stack

    Args:
        UI_stack: illumination array, shape ``(A, H, W)``.
        U_stack:  reconstructed latent field, shape ``(A, H, W)``.
        output_dir: directory to save the .mat file.
        filename: output filename (default ``odt_input.mat``).

    Returns:
        Full path to the saved .mat file.
    """
    from scipy.io import savemat

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    savemat(str(out_path), {
        'Illumination_stack': np.asarray(UI_stack, dtype=np.complex64),
        'Output_stack': np.asarray(U_stack, dtype=np.complex64),
    })

    print(f"Data saved to {out_path} in MATLAB format.")
    return out_path
