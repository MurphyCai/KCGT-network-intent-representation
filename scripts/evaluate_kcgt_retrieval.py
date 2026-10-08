"""Command-line wrapper for KCGT retrieval evaluation."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

runpy.run_path(str(ROOT / "src" / "experiments" / "evaluate_kcgt_retrieval.py"), run_name="__main__")
