"""
config.py
=========
Typed configuration loaded from YAML.

Schema versioning: the top-level key ``schema_version`` (int) is
checked against ``CURRENT_SCHEMA``. Missing key is tolerated (treated
as v1) to keep hand-written configs easy.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

CURRENT_SCHEMA = 1


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OpticsConfig:
    wavelength: float          # micrometres
    NA: float
    n0: float                  # medium refractive index
    scan_step: float            # axial scan step (micrometres) — multiplied by slice indices to get physical distance

    @property
    def k(self) -> float:
        import math
        return 2.0 * math.pi * self.n0 / self.wavelength

    @property
    def npix(self) -> float:
        """Nyquist-matched lateral sample spacing (micrometres)."""
        return self.wavelength / (2.0 * self.NA)


@dataclass
class GridConfig:
    Nx: int
    Ny: int
    Nz: int
    crop_size: Optional[int] = None
    """Central region cropped out of the full ``(Nx,Ny)`` field when
    producing the CASS image. ``None`` defaults to ``Nx//2`` (the
    legacy default when ``Nx == Ny``)."""

    def resolved_crop(self) -> int:
        return self.crop_size if self.crop_size is not None else self.Nx // 2


@dataclass
class DataConfig:
    illumination_file: str                    # .mat or .npy
    illumination_mat_key: Optional[str] = None
    cass_file: str = ""                       # .npy measured CASS images
    cass_amplitude_scale: float = 1.0         # multiply CASS by (usually n_angles)
    distance_range: Tuple[int, int] = (-12, 13)  # slice indices (int) — converted to um via dz
    distance_step: int = 1                    # stride passed to range(start,stop,step)
    pad_to_full_grid: bool = True             # symmetric zero-pad UI_stack from (N0,N0) to (Nx,Ny)
    extend_illumination: bool = False         # extend plane waves to full grid (avoids edge artifacts)
    reverse_cass_z: bool = False              # reverse CASS along z (legacy `[::-1]` use)
    cass_crop: Optional[List[int]] = None     # [y0, y1, x0, x1] — crop measured CASS at load time
    depths_file: Optional[str] = None         # .npy file of depth indices — alternative to distance_range
    depths_center: Optional[float] = None     # reference index subtracted before multiplying by scan_step
    depth_skip_first: int = 0                 # drop the first N depth frames (CASS + depths) in file order before use


@dataclass
class OptConfig:
    n_epochs: int = 30
    batch_size: int = 5
    learning_rate: float = 0.05
    reg_weight: float = 0.01
    loss_fn: str = "MSE"                        # 'MSE' | 'Pearson'
    reg_mode: str = "L2"                      # 'L2' | 'L2+TV' | 'TV' | 'none'
    lr_schedule: str = "none"                 # 'none' | 'cosine'
    checkpoint_every: int = 5
    save_snapshots: bool = True
    snapshot_angles: Optional[List[int]] = None  # angle indices to plot at each checkpoint
    save_crop_size: Optional[int] = None         # crop U_stack before saving; None = same as grid.crop_size
    grad_clip_norm: float = 0.0                   # max gradient norm (0 = no clipping)
    early_stop_patience: int = 0                  # stop after N epochs with < early_stop_min_delta improvement (0 = off)
    early_stop_min_delta: float = 1e-4            # minimum relative improvement to count as progress
    denoise_every: int = 0                       # apply median filter every N epochs (0 = off)
    denoise_kernel: int = 3                      # median filter kernel size (spatial, per angle)


@dataclass
class RuntimeConfig:
    device: str = "cuda"                      # 'cuda' | 'cpu' | 'cuda:0' ...
    seed: int = 0
    use_amp: bool = False
    use_compile: bool = False
    num_workers: int = 0
    pin_memory: bool = True


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"


@dataclass
class IOConfig:
    output_dir: str = ""                      # run results root; empty or relative -> <project>/results/[output_dir]
    run_name: Optional[str] = None            # overridden from CLI; else auto


@dataclass
class Config:
    name: str
    optics: OpticsConfig
    grid: GridConfig
    data: DataConfig
    optimisation: OptConfig
    runtime: RuntimeConfig
    io: IOConfig
    schema_version: int = CURRENT_SCHEMA
    notes: str = ""


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

_ENV_RE = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _expand(obj: Any) -> Any:
    """Recursively expand ``${ENV}`` and ``${ENV:-default}`` in strings."""
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        def sub(m: re.Match) -> str:
            name, default = m.group(1), m.group(2) or ""
            return os.environ.get(name, default)
        return _ENV_RE.sub(sub, obj)
    return obj


def _strict_build(cls, data: Dict[str, Any]):
    """Build a dataclass, rejecting unknown keys with a clear error."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"Unknown keys for {cls.__name__}: {sorted(unknown)}. "
            f"Allowed: {sorted(known)}"
        )
    return cls(**data)


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and materialise into :class:`Config`.

    Env-var expansion: ``${VAR}`` or ``${VAR:-default}`` anywhere in a
    string value. ``tuple`` fields may be written as YAML lists.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = _expand(raw)

    schema = raw.get("schema_version", CURRENT_SCHEMA)
    if schema != CURRENT_SCHEMA:
        raise ValueError(
            f"Config {path} has schema_version={schema}; "
            f"expected {CURRENT_SCHEMA}. Update the config or bump CURRENT_SCHEMA."
        )

    # Normalise tuple-like lists
    data_section = raw.get("data", {})
    if "distance_range" in data_section and isinstance(data_section["distance_range"], list):
        lo, hi = data_section["distance_range"]
        data_section["distance_range"] = (int(lo), int(hi))

    cfg = Config(
        name=raw["name"],
        optics=_strict_build(OpticsConfig, raw["optics"]),
        grid=_strict_build(GridConfig, raw["grid"]),
        data=_strict_build(DataConfig, data_section),
        optimisation=_strict_build(OptConfig, raw.get("optimisation", {})),
        runtime=_strict_build(RuntimeConfig, raw.get("runtime", {})),
        io=_strict_build(IOConfig, raw.get("io", {})),
        schema_version=schema,
        notes=raw.get("notes", ""),
    )
    cfg.io.output_dir = _resolve_output_dir(cfg.io.output_dir)
    return cfg


def _resolve_output_dir(path: str) -> str:
    """Resolve `io.output_dir`: empty or relative -> <project>/results/[path]."""
    p = Path(path).expanduser() if path else Path("")
    if not path or not p.is_absolute():
        return str(DEFAULT_RESULTS_ROOT / p)
    return str(p)


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    """Round-trip to a plain dict (for logging / checkpointing)."""
    return asdict(cfg)
