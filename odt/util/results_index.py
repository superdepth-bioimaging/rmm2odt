"""Connect each run back to its data source via a CSV+MD index.

Port of ``gvas_torch/utils.py`` adapted for BPM_ODT's dict-based config.
The index lives at ``<results_root>/results_index.{csv,md}`` and records
``run_name, config_name, data_source, n_iter, final_cost, elapsed_s, date``
for every reconstruction.
"""
from __future__ import annotations

import csv
import datetime as _dt
import os
from pathlib import Path
from typing import Mapping


def default_run_name(config_name: str, results_root: str | Path) -> str:
    """Compose a unique run identifier.

    Returns ``YYMMDD_<config_name>``; if an entry with that exact name is
    already present in ``<results_root>/results_index.csv``, append a
    ``_THHMM`` timestamp suffix to avoid collision.
    """
    date = _dt.datetime.now().strftime("%y%m%d")
    name = f"{date}_{config_name}"

    csv_path = Path(results_root).expanduser() / "results_index.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("run_name") == name:
                    return f"{name}_{_dt.datetime.now().strftime('T%H%M')}"
    return name


def append_results_index(
    results_root: str | Path,
    run_name: str,
    config_name: str,
    cfg: Mapping,
    final_cost: float,
    elapsed: float,
) -> None:
    """Append one row to ``results_index.csv`` and ``results_index.md``.

    Creates each file with a header on first use.
    """
    results_root = Path(results_root).expanduser()
    results_root.mkdir(parents=True, exist_ok=True)

    data_source = str(cfg.get("io", {}).get("data_path", ""))
    n_iter = cfg.get("optim", {}).get("n_iter", "")
    now = _dt.datetime.now()

    row = {
        "run_name":    run_name,
        "config_name": config_name,
        "data_source": data_source,
        "n_iter":      n_iter,
        "final_cost":  f"{final_cost:.6e}",
        "elapsed_s":   f"{elapsed:.1f}",
        "date":        now.strftime("%Y-%m-%d %H:%M"),
    }

    csv_path = results_root / "results_index.csv"
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    md_path = results_root / "results_index.md"
    md_exists = md_path.exists()
    data_label = Path(data_source).parent.name if data_source else ""
    with md_path.open("a", encoding="utf-8") as f:
        if not md_exists:
            f.write("# BPM_ODT Results Index\n\n")
            f.write("| Run | Config | Data | Iters | Final cost | Time | Date |\n")
            f.write("|-----|--------|------|-------|------------|------|------|\n")
        f.write(
            f"| {run_name} | {config_name} | `{data_label}` "
            f"| {n_iter} | {final_cost:.4e} | {elapsed:.0f}s "
            f"| {now.strftime('%Y-%m-%d')} |\n"
        )
