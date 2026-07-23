"""lacuna — research-frontier forecasting over the arXiv embedding cloud.

Placeholder name (final name deferred). v2 of arxiv-heatmap: forecast which
regions of the SPECTER2 embedding cloud are emerging vs saturating, headlined
by a pre-registered, multi-cutoff out-of-time harness against a
smoothed-momentum baseline.

Pipeline order: ingest -> embed -> reduce -> regions -> series -> forecast
-> emergence -> validate -> labels -> export_map / export_figs.
"""
from __future__ import annotations

__version__ = "0.0.0"

from . import config  # noqa: F401
