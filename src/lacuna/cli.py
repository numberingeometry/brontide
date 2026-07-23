"""Shared CLI plumbing for the stage runners.

Every stage supports ``--sample N`` so the storage-limited CPU-only environment can smoke-test
the whole pipeline before the host full run.
"""
from __future__ import annotations

import argparse


def sample_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--sample", type=int, default=None,
                   metavar="N", help="run on the first N papers (smoke test); omit for full")
    return p


def parse_sample(description: str) -> int | None:
    return sample_parser(description).parse_args().sample
