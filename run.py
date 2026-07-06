#!/usr/bin/env python
"""rmm2odt — single end-to-end CLI: reflection matrix -> 3-D ODT image.

    python run.py --config configs/example_pipeline.yaml
    python run.py --config configs/example_pipeline.yaml --dry-run
    python run.py --config configs/my.yaml --from gvas --to odt

Stages (in order): multilayer | gvas | odt | render
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the vendored top-level packages (multilayer_torch, gvas_torch, odt)
# and the glue modules (rmm_*, _common) importable.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rmm_config import STAGES   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="rmm2odt: reflection matrix -> multilayer -> GVAS -> BPM_ODT -> image")
    ap.add_argument("--config", required=True, metavar="PATH",
                    help="unified pipeline YAML (see configs/example_pipeline.yaml)")
    ap.add_argument("--from", dest="from_stage", choices=STAGES, default=None,
                    help="first stage to run (resume). default: multilayer")
    ap.add_argument("--to", dest="to_stage", choices=STAGES, default=None,
                    help="last stage to run. default: render")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate config + print stage plan and I/O paths without executing")
    args = ap.parse_args(argv)

    from rmm_pipeline import run_pipeline
    run_pipeline(args.config, args.from_stage, args.to_stage, dry=args.dry_run)


if __name__ == "__main__":
    main()
