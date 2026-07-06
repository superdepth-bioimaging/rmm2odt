"""
gvas_torch — Generative Virtual Angular-Scan reconstruction (PyTorch).

Refactored from to_virtual_angular_scan_model_torch_legacy.
Physics reference: ../theoretical_foundation.pdf
"""
from . import physics  # noqa: F401
from . import config as _config  # noqa: F401
from . import loss  # noqa: F401

__version__ = "0.1.0"
