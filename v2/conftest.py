"""Make standalone v2 experiment packages importable under root pytest."""

from __future__ import annotations

import sys
from pathlib import Path


V2 = Path(__file__).resolve().parent
for directory in (
    V2,
    V2 / "incremental_decode",
    V2 / "speculation_traces",
    V2 / "human_annotation",
):
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)
