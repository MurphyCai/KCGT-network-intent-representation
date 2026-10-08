"""Command-line wrapper for baseline classification training."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

runpy.run_path(str(ROOT / "src" / "experiments" / "train_baseline_classification.py"), run_name="__main__")
