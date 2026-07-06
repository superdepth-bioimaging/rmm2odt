"""Sanity checks for reconstruction config dicts. Port of validate_config.m."""
from __future__ import annotations

from typing import Any, Mapping


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Validate a reconstruction config dict.

    Raises ``ValueError`` (with a descriptive message) on missing fields or
    out-of-range values. Mirrors the MATLAB ``validate_config.m`` behavior.
    """
    _require_section(cfg, "physics")
    for f in ("wl", "n0", "NA", "npix", "dz"):
        _require_field(cfg["physics"], f"physics.{f}")

    _require_section(cfg, "geom")
    for f in ("Nx", "Ny", "Nz"):
        _require_field(cfg["geom"], f"geom.{f}")

    _require_section(cfg, "optim")
    for f in ("n_iter", "gamma", "damping", "ang_num", "stochastic"):
        _require_field(cfg["optim"], f"optim.{f}")

    _require_section(cfg, "reg")
    for f in ("tau", "n_iter_tv", "n_upper"):
        _require_field(cfg["reg"], f"reg.{f}")

    _require_section(cfg, "io")
    for f in ("data_path", "output_dir"):
        _require_field(cfg["io"], f"io.{f}")

    # Range checks
    if cfg["physics"]["wl"] <= 0:    raise ValueError("physics.wl must be > 0")
    if cfg["physics"]["n0"] <= 0:    raise ValueError("physics.n0 must be > 0")
    if cfg["physics"]["NA"] <= 0:    raise ValueError("physics.NA must be > 0")
    if cfg["geom"]["Nx"] <= 0:       raise ValueError("geom.Nx must be > 0")
    if cfg["geom"]["Ny"] <= 0:       raise ValueError("geom.Ny must be > 0")
    if cfg["geom"]["Nz"] <= 0:       raise ValueError("geom.Nz must be > 0")
    if cfg["optim"]["n_iter"] < 1:   raise ValueError("optim.n_iter must be >= 1")
    if cfg["optim"]["gamma"] <= 0:   raise ValueError("optim.gamma must be > 0")
    if cfg["optim"]["ang_num"] < 1:  raise ValueError("optim.ang_num must be >= 1")
    if cfg["reg"]["tau"] < 0:        raise ValueError("reg.tau must be >= 0")
    if cfg["reg"]["n_iter_tv"] < 1:  raise ValueError("reg.n_iter_tv must be >= 1")

    # Optional reg fields
    n_lower = cfg["reg"].get("n_lower")
    if n_lower is not None and n_lower > cfg["reg"]["n_upper"]:
        raise ValueError("reg.n_lower must be <= reg.n_upper")

    # Dimension parity (BPM kernel symmetry assumes even sizes)
    if cfg["geom"]["Nx"] % 2 or cfg["geom"]["Ny"] % 2 or cfg["geom"]["Nz"] % 2:
        import warnings
        warnings.warn(
            f"Volume dimensions should be even (Nx={cfg['geom']['Nx']}, "
            f"Ny={cfg['geom']['Ny']}, Nz={cfg['geom']['Nz']}).",
            stacklevel=2,
        )

    # Optional preprocess fields
    pp = cfg.get("preprocess", {})
    win = pp.get("window", "none")
    if win.lower() not in ("none", "hann", "tukey"):
        raise ValueError(
            f"preprocess.window must be 'none', 'hann', or 'tukey'; got {win!r}"
        )
    pr = pp.get("phase_ramp_remove", {})
    if isinstance(pr, dict):
        roi = pr.get("roi")
        if roi is not None and len(roi) > 0 and len(roi) != 4:
            raise ValueError(
                "preprocess.phase_ramp_remove.roi must be [y1, y2, x1, x2] "
                f"(4 elements); got {len(roi)}"
            )


def _require_section(cfg: Mapping[str, Any], name: str) -> None:
    if name not in cfg:
        raise ValueError(f"Missing required config section: {name}")


def _require_field(section: Mapping[str, Any], name: str) -> None:
    leaf = name.split(".")[-1]
    if leaf not in section:
        raise ValueError(f"Missing required config field: {name}")
