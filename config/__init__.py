from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "v6_rfactor" / "annual"
OUTPUT_DIR = ROOT / "output"
DOCS_DIR = ROOT / "docs"

__all__ = ["ROOT", "DATA_DIR", "OUTPUT_DIR", "DOCS_DIR"]
