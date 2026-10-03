"""Make the vendored method code importable as ``src.*``.

The experiment repo ships the EquiCEval engine and the RA baseline under
``vendor/src`` (a manual vendoring of the two method repos). Importing the
``expeval`` package registers ``vendor`` on ``sys.path`` so ``src.equiceval`` and
``src.baseline`` resolve without setting PYTHONPATH manually.
"""
import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[1] / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
