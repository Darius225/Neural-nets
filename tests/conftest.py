"""Shared pytest config — adds the repo root to sys.path so tests can
import from ``src.*`` without needing an editable install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
